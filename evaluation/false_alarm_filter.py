import os, sys, re, time, json, math, random

IN_COLAB = "google.colab" in sys.modules
IN_KAGGLE = os.path.exists("/kaggle/working")
SMOKE = os.environ.get("PCB_SMOKE", "0") == "1"

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    PROJECT_DIR = "/content/drive/MyDrive/PCB_Defect_Detection"
elif IN_KAGGLE:
    PROJECT_DIR = "/kaggle/working"
else:
    PROJECT_DIR = os.environ.get("PCB_PROJECT_DIR", os.getcwd())
os.makedirs(PROJECT_DIR, exist_ok=True)
DATA_TMP = "/content" if IN_COLAB else ("/tmp" if IN_KAGGLE else PROJECT_DIR)

if IN_KAGGLE:
    import shutil
    def _run_folder(path):
        parts = path.replace("\\", "/").split("/")
        for d in ("runs_detector", "runs_deblur_binarize", "runs_pku"):
            if d in parts: return d
        name = parts[-1]
        if name.startswith(("rgdb", "deblur_", "fa_filter")): return "runs_deblur_binarize"
        if name.startswith("pku_"): return "runs_pku"
        if name.startswith(("ref_", "testonly_")): return "runs_detector"
        return None
    _copied = []
    for root, _, files in os.walk("/kaggle/input"):
        if "PCBData" in root or "PCB_USED" in root or "Annotations" in root:
            continue
        for fn in files:
            if not fn.endswith((".pth", ".json")): continue
            src = os.path.join(root, fn); d = _run_folder(src)
            if d is None: continue
            dst = os.path.join(PROJECT_DIR, d, fn)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst); _copied.append(f"{d}/{fn}")
    print("copied from /kaggle/input:", _copied if _copied else "nothing (no checkpoint dataset attached)")

DEEPPCB_DIR = os.path.join(DATA_TMP, "DeepPCB")
if not os.path.exists(DEEPPCB_DIR):
    os.system(f"git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git '{DEEPPCB_DIR}'")
    assert os.path.exists(DEEPPCB_DIR), "DeepPCB download failed. On Kaggle: Settings > Internet ON (needs a verified phone number)."

OUT_DIR = os.path.join(PROJECT_DIR, "runs_detector")
os.makedirs(OUT_DIR, exist_ok=True)
print("DeepPCB:", DEEPPCB_DIR, "| outputs:", OUT_DIR)


import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, box_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| device", device)


CFG = dict(
    split="official",
    use_template=True,
    use_nnvr=True,
    pretrained=True,
    epochs=20,
    batch_size=2,
    lr=0.005,
    lr_steps=(12, 16),
    weight_decay=1e-4,
    train_blur_prob=0.0,
    num_workers=2 if (IN_COLAB or IN_KAGGLE) else 0,
    seed=42,
)
if SMOKE:
    CFG.update(epochs=1, pretrained=False, num_workers=0)

CLASS_NAMES = ["background", "open", "short", "mousebite", "spur", "copper", "pin-hole"]
NUM_CLASSES = len(CLASS_NAMES)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
seed_everything(CFG["seed"])


MOTION_LEN = [0, 5, 9, 13, 17, 23]
DEFOCUS_R = [0, 1, 2, 3, 4, 5]
NOISE_STD = [0, 2, 3, 4, 5, 6]

def motion_kernel(length, angle):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    return k / max(k.sum(), 1e-6)

def defocus_kernel(radius):
    size = 2 * radius + 1
    k = np.zeros((size, size), np.float32)
    cv2.circle(k, (radius, radius), radius, 1.0, -1)
    return k / k.sum()

def degrade(img, severity, rng, kind="mixed"):
    if severity <= 0:
        return img
    if kind == "mixed":
        kind = "motion" if rng.random() < 0.5 else "defocus"
    if kind == "motion":
        k = motion_kernel(MOTION_LEN[severity], rng.uniform(0, 180))
    else:
        k = defocus_kernel(DEFOCUS_R[severity])
    out = cv2.filter2D(img.astype(np.float32), -1, k)
    out += rng.normal(0, NOISE_STD[severity], out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def read_pairs(root, split_file):
    base = os.path.join(root, "PCBData")
    pairs = []
    with open(os.path.join(base, split_file)) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            stem = os.path.splitext(parts[0])[0]
            pairs.append((os.path.join(base, stem + "_test.jpg"),
                          os.path.join(base, stem + "_temp.jpg"),
                          os.path.join(base, parts[1])))
    return pairs

def read_boxes(ann_path):
    boxes, labels = [], []
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            for line in f:
                vals = [int(float(v)) for v in re.split(r"[,\s]+", line.strip()) if v]
                if len(vals) == 5 and vals[4] > 0:
                    x1, y1, x2, y2, c = vals
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2]); labels.append(c)
    return (np.array(boxes, np.float32).reshape(-1, 4), np.array(labels, np.int64))

def make_splits(root, mode, seed=42):
    rng = random.Random(seed)
    trainval, test = read_pairs(root, "trainval.txt"), read_pairs(root, "test.txt")
    if mode == "official":
        rng.shuffle(trainval)
        n_val = len(trainval) // 10
        return trainval[n_val:], trainval[:n_val], test
    if mode == "mgre_811":
        allp = trainval + test
        rng.shuffle(allp)
        n = len(allp); a, b = int(0.8 * n), int(0.9 * n)
        return allp[:a], allp[a:b], allp[b:]
    raise ValueError(mode)

class DeepPCBPairs(Dataset):
    def __init__(self, pairs, train=False, blur_prob=0.0, eval_severity=0, seed=0):
        self.pairs, self.train = pairs, train
        self.blur_prob, self.eval_severity, self.seed = blur_prob, eval_severity, seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        tp, rp, ap = self.pairs[i]
        test = cv2.cvtColor(cv2.imread(tp), cv2.COLOR_BGR2RGB)
        temp = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
        boxes, labels = read_boxes(ap)
        h, w = test.shape[:2]

        if self.train:
            rng = np.random.default_rng()
            if rng.random() < self.blur_prob:
                test = degrade(test, int(rng.integers(1, 6)), rng)
            if rng.random() < 0.5:
                test, temp = test[:, ::-1], temp[:, ::-1]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if rng.random() < 0.5:
                test, temp = test[::-1], temp[::-1]
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        elif self.eval_severity > 0:
            test = degrade(test, self.eval_severity, np.random.default_rng(self.seed * 100003 + i))

        x = np.concatenate([test, temp], axis=2).astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
        target = {"boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels),
                  "image_id": torch.tensor(i)}
        return x, target

def collate(batch):
    return tuple(zip(*batch))

train_pairs, val_pairs, test_pairs = make_splits(DEEPPCB_DIR, CFG["split"], CFG["seed"])
if SMOKE:
    train_pairs, val_pairs, test_pairs = train_pairs[:8], val_pairs[:4], test_pairs[:6]
print(f"split={CFG['split']}: train {len(train_pairs)} | val {len(val_pairs)} | test {len(test_pairs)}")

x0, t0 = DeepPCBPairs(train_pairs)[0]
print("sample:", tuple(x0.shape), "boxes", tuple(t0["boxes"].shape), "labels", t0["labels"].tolist())


class RefGuidedBackbone(nn.Module):
    def __init__(self, pretrained=True, use_template=True, trainable_layers=3):
        super().__init__()
        self.body = resnet_fpn_backbone(backbone_name="resnet50",
                                        weights=ResNet50_Weights.DEFAULT if pretrained else None,
                                        trainable_layers=trainable_layers)
        self.out_channels = 256
        self.use_template = use_template
        if use_template:
            self.fuse = nn.ModuleDict({
                k: nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 1))
                for k in ["0", "1", "2", "3", "pool"]})
            for m in self.fuse.values():
                nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)

    def forward(self, x):
        test, temp = x[:, :3], x[:, 3:]
        if not self.use_template:
            return self.body(test)
        feats = self.body(torch.cat([test, temp], dim=0))
        b = test.shape[0]
        out = {}
        for k, f in feats.items():
            ft, fr = f[:b], f[b:]
            out[k] = ft + self.fuse[k](torch.cat([ft, (ft - fr).abs()], dim=1))
        return out

class NNVR(nn.Module):
    def __init__(self, dim=1024, latent=256, top_k=8):
        super().__init__()
        self.top_k = top_k
        self.proj = nn.Sequential(nn.Linear(dim, latent), nn.ReLU(inplace=True), nn.Linear(latent, latent))
        self.gcn1, self.gcn2 = nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x):
        n = x.shape[0]
        if n <= 1:
            return x
        z = F.normalize(self.proj(x), dim=-1)
        sim = z @ z.t()
        val, idx = sim.topk(min(self.top_k, n), dim=-1)
        adj = torch.zeros_like(sim).scatter_(-1, idx, val.clamp(min=0))
        adj = adj / (adj.sum(-1, keepdim=True) + 1e-6)
        h = F.relu(self.gcn1(adj @ x))
        h = F.relu(self.gcn2(adj @ h))
        return x + h

class CountingRoIAlign(MultiScaleRoIAlign):
    def forward(self, x, boxes, image_shapes):
        self.last_counts = [len(b) for b in boxes]
        return super().forward(x, boxes, image_shapes)

class NNVRBoxHead(nn.Module):
    def __init__(self, pooler, in_ch=256 * 7 * 7, rep=1024, use_nnvr=True):
        super().__init__()
        self.pooler = pooler
        self.mlp = TwoMLPHead(in_ch, rep)
        self.nnvr = NNVR(rep) if use_nnvr else None

    def forward(self, x):
        x = self.mlp(x)
        if self.nnvr is None:
            return x
        chunks = x.split(self.pooler.last_counts, dim=0)
        return torch.cat([self.nnvr(c) for c in chunks], dim=0)

def build_model(cfg):
    backbone = RefGuidedBackbone(cfg["pretrained"], cfg["use_template"])
    anchors = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    pooler = CountingRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    head = NNVRBoxHead(pooler, use_nnvr=cfg["use_nnvr"])
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return FasterRCNN(backbone, rpn_anchor_generator=anchors,
                      box_roi_pool=pooler, box_head=head, box_predictor=FastRCNNPredictor(1024, NUM_CLASSES),
                      min_size=640, max_size=640, image_mean=mean * 2, image_std=std * 2,
                      box_detections_per_img=100)


def ap_voc(rec, prec):
    mrec = np.concatenate([[0.0], rec, [1.0]]); mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

def evaluate_detections(preds, gts, iou_thr=0.5, score_thr=0.5):
    aps, tp_all, fp_all, n_gt_all = {}, 0, 0, 0
    for c in range(1, NUM_CLASSES):
        dets, n_gt, gt_c = [], 0, []
        for i, (p, g) in enumerate(zip(preds, gts)):
            gb = g["boxes"][g["labels"] == c]
            gt_c.append((gb, np.zeros(len(gb), bool)))
            n_gt += len(gb)
            m = p["labels"] == c
            for b, s in zip(p["boxes"][m], p["scores"][m]):
                dets.append((float(s), i, b))
        dets.sort(key=lambda d: -d[0])
        tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
        for k, (s, i, b) in enumerate(dets):
            gb, used = gt_c[i]
            if len(gb):
                ious = box_iou(b[None], gb)[0].numpy()
                j = int(ious.argmax())
                if ious[j] >= iou_thr and not used[j]:
                    used[j] = True; tp[k] = 1
                    continue
            fp[k] = 1
        if n_gt == 0:
            continue
        ctp, cfp = np.cumsum(tp), np.cumsum(fp)
        rec = ctp / n_gt
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        aps[CLASS_NAMES[c]] = ap_voc(rec, prec) if len(dets) else 0.0
        keep = np.array([d[0] >= score_thr for d in dets], bool)
        tp_all += int(tp[keep].sum()); fp_all += int(fp[keep].sum()); n_gt_all += n_gt
    P = tp_all / max(tp_all + fp_all, 1); R = tp_all / max(n_gt_all, 1)
    return {"mAP": float(np.mean(list(aps.values()))) if aps else 0.0, "AP": aps,
            "precision": P, "recall": R, "f1": 2 * P * R / max(P + R, 1e-9)}

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, gts, t_model, n_img = [], [], 0.0, 0
    for xs, ts in loader:
        xs = [x.to(device) for x in xs]
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        outs = model(xs)
        if device.type == "cuda": torch.cuda.synchronize()
        t_model += time.time() - t0; n_img += len(xs)
        preds += [{k: v.cpu() for k, v in o.items()} for o in outs]
        gts += [{"boxes": t["boxes"], "labels": t["labels"]} for t in ts]
    return preds, gts, n_img / max(t_model, 1e-9)

def full_report(preds, gts, fps=None):
    r50 = evaluate_detections(preds, gts, 0.5)
    r33 = evaluate_detections(preds, gts, 0.33)
    coco = float(np.mean([evaluate_detections(preds, gts, t)["mAP"] for t in np.arange(0.5, 0.96, 0.05)]))
    rep = {"mAP@0.5": r50["mAP"], "mAP@0.5:0.95": coco, "AP@0.5 per class": r50["AP"],
           "P@0.5": r50["precision"], "R@0.5": r50["recall"], "F1@0.5": r50["f1"],
           "mAP@0.33 (DeepPCB protocol)": r33["mAP"], "F-score@0.33 (DeepPCB protocol)": r33["f1"]}
    if fps is not None:
        rep["FPS"] = fps
    return rep

def count_gflops(model, size=640):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval()
    x = [torch.rand(6, size, size, device=device)]
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    return fc.get_total_flops() / 2 / 1e9

def make_loader(pairs, train=False, severity=0, shuffle=False):
    ds = DeepPCBPairs(pairs, train=train, blur_prob=CFG["train_blur_prob"] if train else 0.0,
                      eval_severity=severity, seed=CFG["seed"])
    return DataLoader(ds, batch_size=CFG["batch_size"], shuffle=shuffle, num_workers=CFG["num_workers"],
                      collate_fn=collate, pin_memory=device.type == "cuda")


OUT_DIR = os.path.join(PROJECT_DIR, "runs_deblur_binarize")
os.makedirs(OUT_DIR, exist_ok=True)

RG = dict(
    in_template=True,
    base_ch=32,
    epochs_a=15, lr_a=1e-3, batch_a=4,
    epochs_b=6, lr_b_det=1e-3, lr_b_rg=2e-4, batch_b=2, alpha_b=1.0,
    clean_prob=0.2,
    box_weight=10.0, diff_weight=5.0, dice_weight=1.0, edge_weight=1.0,
    eval_severities=[0, 3, 4, 5],
    run_stage_b=True,
    stage_b_use_saved_best=True,
    det_ckpt=os.path.join(PROJECT_DIR, "runs_detector", "ref_nnvr_official_best.pth"),
)
if SMOKE:
    RG.update(epochs_a=1, epochs_b=1, batch_a=2, eval_severities=[0, 3])
TAG = "rgdb" if RG["in_template"] else "deblur_notemplate"
print("outputs ->", OUT_DIR, "| detector checkpoint:", RG["det_ckpt"], "exists:", os.path.exists(RG["det_ckpt"]))


class RGPairs(Dataset):
    def __init__(self, pairs, train=False, severity=0, seed=0, clean_prob=0.2):
        self.pairs, self.train, self.severity, self.seed, self.clean_prob = pairs, train, severity, seed, clean_prob

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        tp, rp, ap = self.pairs[i]
        test = cv2.cvtColor(cv2.imread(tp), cv2.COLOR_BGR2RGB)
        temp = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
        boxes, labels = read_boxes(ap)
        h, w = test.shape[:2]
        if self.train:
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                test, temp = test[:, ::-1], temp[:, ::-1]; boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if rng.random() < 0.5:
                test, temp = test[::-1], temp[::-1]; boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            sev = 0 if rng.random() < self.clean_prob else int(rng.integers(1, 6))
        else:
            sev, rng = self.severity, np.random.default_rng(self.seed * 100003 + i)
        test = np.ascontiguousarray(test); temp = np.ascontiguousarray(temp)
        blur = degrade(test, sev, rng) if sev > 0 else test
        f = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
        blur3, test3, temp3 = f(blur), f(test), f(temp)
        return {"blur3": blur3, "blur": blur3.mean(0, keepdim=True), "temp3": temp3,
                "temp": temp3.mean(0, keepdim=True), "clean": (test3.mean(0, keepdim=True) > 0.5).float(),
                "boxes": torch.from_numpy(boxes), "labels": torch.from_numpy(labels), "sev": sev}

def rg_collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ["blur3", "blur", "temp3", "temp", "clean"]}
    out.update(boxes=[b["boxes"] for b in batch], labels=[b["labels"] for b in batch],
               sev=[b["sev"] for b in batch])
    return out

_SX = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
def sobel(x):
    sx, sy = _SX.to(x), _SX.transpose(2, 3).to(x)
    gx, gy = F.conv2d(x, sx, padding=1), F.conv2d(x, sy, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(),
                                  nn.Conv2d(c, c, 3, padding=1), nn.GroupNorm(8, c))
    def forward(self, x):
        return F.gelu(x + self.body(x))

class RGDBNet(nn.Module):
    def __init__(self, in_template=True, c=32):
        super().__init__()
        self.in_template = in_template
        cin = 4 if in_template else 2
        self.stem = nn.Sequential(nn.Conv2d(cin, c, 3, padding=1), ResBlock(c))
        self.d1 = nn.Sequential(nn.Conv2d(c, 2 * c, 3, stride=2, padding=1), ResBlock(2 * c), ResBlock(2 * c))
        self.d2 = nn.Sequential(nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1), ResBlock(4 * c), ResBlock(4 * c))
        self.d3 = nn.Sequential(nn.Conv2d(4 * c, 8 * c, 3, stride=2, padding=1), ResBlock(8 * c), ResBlock(8 * c))
        self.u3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2)
        self.m3 = nn.Sequential(nn.Conv2d(8 * c, 4 * c, 3, padding=1), ResBlock(4 * c))
        self.u2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2)
        self.m2 = nn.Sequential(nn.Conv2d(4 * c, 2 * c, 3, padding=1), ResBlock(2 * c))
        self.u1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.m1 = nn.Sequential(nn.Conv2d(2 * c, c, 3, padding=1), ResBlock(c))
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, blur, temp):
        parts = [blur - 0.5, sobel(blur)]
        if self.in_template:
            parts += [temp - 0.5, sobel(temp)]
        e0 = self.stem(torch.cat(parts, 1)); e1 = self.d1(e0); e2 = self.d2(e1); e3 = self.d3(e2)
        x = self.m3(torch.cat([self.u3(e3), e2], 1))
        x = self.m2(torch.cat([self.u2(x), e1], 1))
        x = self.m1(torch.cat([self.u1(x), e0], 1))
        return self.head(x) + 8.0 * (blur - 0.5)

def box_mask(boxes_list, shape, device):
    m = torch.zeros(shape, device=device)
    for b, boxes in enumerate(boxes_list):
        for x1, y1, x2, y2 in boxes.round().long().tolist():
            m[b, 0, max(y1, 0):y2, max(x1, 0):x2] = 1.0
    return m

def rgdb_loss(logits, batch, cfg=RG):
    clean, temp = batch["clean"], (batch["temp"] > 0.5).float()
    p = torch.sigmoid(logits.float())
    diff_gt = (clean - temp).abs()
    w = 1.0 + cfg["box_weight"] * box_mask(batch["boxes"], clean.shape, clean.device) + cfg["diff_weight"] * diff_gt
    bce = (F.binary_cross_entropy_with_logits(logits.float(), clean, reduction="none") * w).sum() / w.sum()
    fg_p, fg_c = 1 - p, 1 - clean
    dice = 1 - (2 * (fg_p * fg_c).sum() + 1) / (fg_p.sum() + fg_c.sum() + 1)
    edge = (sobel(p) - sobel(clean)).abs().mean()
    diff = ((p - temp).abs() - diff_gt).abs().mean()
    parts = {"bce": bce, "dice": dice, "edge": edge, "diff": diff}
    total = bce + cfg["dice_weight"] * dice + cfg["edge_weight"] * edge + diff
    return total, parts

def _gauss(win=11, sigma=1.5):
    g = torch.exp(-(torch.arange(win) - win // 2) ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, win, win)

def ssim(a, b):
    g = _gauss().to(a); C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = F.conv2d(a, g, padding=5), F.conv2d(b, g, padding=5)
    va = F.conv2d(a * a, g, padding=5) - mu_a ** 2; vb = F.conv2d(b * b, g, padding=5) - mu_b ** 2
    cov = F.conv2d(a * b, g, padding=5) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return s.mean(dim=(1, 2, 3))

def restoration_metrics(p, batch):
    clean = batch["clean"]
    mse = ((p - clean) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
    pb, cf = (p < 0.5).float(), 1 - clean
    tp = (pb * cf).sum(dim=(1, 2, 3)); f1 = 2 * tp / (pb.sum(dim=(1, 2, 3)) + cf.sum(dim=(1, 2, 3)) + 1e-6)
    m = box_mask(batch["boxes"], clean.shape, clean.device)
    dpa = (((p > 0.5).float() == clean).float() * m).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3)).clamp(min=1)
    return {"PSNR": (10 * torch.log10(1 / mse)).tolist(), "SSIM": ssim(p, clean).tolist(),
            "pixel_F1": f1.tolist(), "DPA": dpa.tolist()}

def otsu_binarize(blur):
    out = []
    for g in blur[:, 0].cpu().numpy():
        g8 = (g * 255).astype(np.uint8)
        _, b = cv2.threshold(cv2.GaussianBlur(g8, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out.append(torch.from_numpy(b.astype(np.float32) / 255.0))
    return torch.stack(out)[:, None].to(blur.device)

def laplacian_var(blur):
    return [float(cv2.Laplacian((g * 255).astype(np.uint8), cv2.CV_64F).var()) for g in blur[:, 0].cpu().numpy()]

rg_net = RGDBNet(RG["in_template"], RG["base_ch"]).to(device)
print(f"RG-DB params: {sum(p.numel() for p in rg_net.parameters()) / 1e6:.2f} M")


def rg_loader(pairs, train=False, severity=0, bs=2, shuffle=False):
    return DataLoader(RGPairs(pairs, train=train, severity=severity, seed=CFG["seed"], clean_prob=RG["clean_prob"]),
                      batch_size=bs, shuffle=shuffle, num_workers=CFG["num_workers"], collate_fn=rg_collate)

def to_dev(batch):
    return {k: (v.to(device) if torch.is_tensor(v) else ([t.to(device) for t in v] if k in ("boxes", "labels") else v))
            for k, v in batch.items()}

def gate_threshold(pairs):
    vals = []
    for batch in rg_loader(pairs, severity=0): vals += laplacian_var(batch["blur"])
    return float(np.percentile(vals, 5))

def load_det_from(state_dict, cfg):
    d = build_model({**cfg, "pretrained": False}).to(device)
    d.load_state_dict(state_dict); d.eval(); d.roi_heads.score_thresh = 0.01
    return d

det_state = torch.load(RG["det_ckpt"], map_location=device, weights_only=False)
det_plain = load_det_from(det_state["model"], det_state["cfg"])

st_joint = torch.load(os.path.join(OUT_DIR, "rgdb_joint.pth"), map_location=device, weights_only=False)
det_joint = load_det_from(st_joint["det"], st_joint["det_cfg"])
rg_net.load_state_dict(st_joint["rg"]); rg_net.eval()
GATE_THR = gate_threshold(val_pairs[:50])
print("models loaded on", device, "| blur gate threshold:", round(GATE_THR, 1))

_fpath = os.path.join(OUT_DIR, "fa_filter.json")
FILTER = json.load(open(_fpath)) if os.path.exists(_fpath) else None
print("template check:", FILTER if FILTER else "not tuned yet (run pcb_filter_eval.ipynb) -> showing raw detections")

_KER = np.ones((3, 3), np.uint8)
def template_check(out, restored, temp, min_px, pad=2):
    d = cv2.morphologyEx(((restored > 0.5) != (temp > 0.5)).astype(np.uint8), cv2.MORPH_OPEN, _KER)
    ii = cv2.integral(d); H, W = d.shape; keep = []
    for x1, y1, x2, y2 in out["boxes"].round().int().tolist():
        x1, y1, x2, y2 = max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, W), min(y2 + pad, H)
        keep.append(x2 > x1 and y2 > y1 and ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1] >= min_px)
    keep = torch.tensor(keep, dtype=torch.bool, device=out["boxes"].device)
    return {k: v[keep] for k, v in out.items()}


KER = np.ones((3, 3), np.uint8)

def diff_counts(rest_bin, temp_bin, boxes, pad=2):
    d = cv2.morphologyEx((rest_bin != temp_bin).astype(np.uint8), cv2.MORPH_OPEN, KER)
    ii = cv2.integral(d)
    H, W = d.shape
    out = []
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
        x2, y2 = min(int(x2) + pad, W), min(int(y2) + pad, H)
        out.append(int(ii[y2, x2] - ii[y1, x2] - ii[y2, x1] + ii[y1, x1]) if x2 > x1 and y2 > y1 else 0)
    return np.array(out, dtype=np.int64)

@torch.no_grad()
def collect(pairs, severity, mode="ours"):
    rows, use_amp = [], device.type == "cuda"
    for batch in rg_loader(pairs, severity=severity):
        batch = to_dev(batch)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            if mode == "ours":
                sharp = torch.tensor([v >= GATE_THR for v in laplacian_var(batch["blur"])], device=device)
                p = batch["blur"].clone()
                todo = (~sharp).nonzero().squeeze(1)
                if len(todo):
                    p[todo] = torch.sigmoid(rg_net(batch["blur"][todo], batch["temp"][todo]).float())
                outs = det_joint(list(torch.cat([p.repeat(1, 3, 1, 1), batch["temp3"]], 1)))
            else:
                p = batch["blur"]
                outs = det_plain(list(torch.cat([batch["blur3"], batch["temp3"]], 1)))
        rb = (p[:, 0].float() > 0.5).cpu().numpy()
        tb = (batch["temp"][:, 0] > 0.5).cpu().numpy()
        for i, o in enumerate(outs):
            boxes = o["boxes"].float().cpu()
            rows.append({"boxes": boxes, "scores": o["scores"].float().cpu(), "labels": o["labels"].cpu(),
                         "diff": torch.from_numpy(diff_counts(rb[i], tb[i], boxes.round().int().tolist())),
                         "gt_boxes": batch["boxes"][i].cpu(), "gt_labels": batch["labels"][i].cpu(),
                         "gt_diff": diff_counts(rb[i], tb[i], batch["boxes"][i].cpu().round().int().tolist())})
    return rows

def prf(rows, score_thr=0.5, min_px=0):
    tp = fp = n_gt = 0
    for r in rows:
        keep = (r["scores"] >= score_thr) & (r["diff"] >= min_px)
        order = r["scores"][keep].argsort(descending=True)
        pb, pl = r["boxes"][keep][order], r["labels"][keep][order]
        gb, gl = r["gt_boxes"], r["gt_labels"]
        used = torch.zeros(len(gb), dtype=torch.bool)
        n_gt += len(gb)
        if len(pb) and len(gb):
            ious = box_iou(pb, gb)
            ious[pl[:, None] != gl[None, :]] = 0
            for k in range(len(pb)):
                cand = ious[k].clone(); cand[used] = 0
                j = int(cand.argmax())
                if cand[j] >= 0.5:
                    used[j] = True; tp += 1
                else:
                    fp += 1
        else:
            fp += len(pb)
    P = tp / max(tp + fp, 1); R = tp / max(n_gt, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)

def filtered_map(rows, min_px):
    preds = [{"boxes": r["boxes"][r["diff"] >= min_px], "scores": r["scores"][r["diff"] >= min_px],
              "labels": r["labels"][r["diff"] >= min_px]} for r in rows]
    gts = [{"boxes": r["gt_boxes"], "labels": r["gt_labels"]} for r in rows]
    return evaluate_detections(preds, gts, 0.5)["mAP"]


SEVS = [0, 3, 4, 5] if not SMOKE else [0, 3]
val_rows = {s: collect(val_pairs, s, "ours") for s in SEVS}

gt_diff_all = np.concatenate([r["gt_diff"] for s in SEVS for r in val_rows[s]])
print("real defects (val): median differing pixels inside the box =", int(np.median(gt_diff_all)))

best = None
for min_px in [0, 2, 4, 6, 8, 12, 16, 24, 32]:
    for thr in np.round(np.arange(0.30, 0.951, 0.05), 2):
        f1 = float(np.mean([prf(val_rows[s], thr, min_px)[2] for s in SEVS]))
        if best is None or f1 > best[0]:
            best = (f1, float(thr), int(min_px))
FILTER = {"score_thr": best[1], "min_px": best[2]}
print(f"chosen on validation: score cut-off {FILTER['score_thr']}, at least {FILTER['min_px']} differing pixels "
      f"(mean val F1 {best[0]:.4f})")
print(f"share of real val defects that pass the template check: {np.mean(gt_diff_all >= FILTER['min_px']):.1%}")
json.dump(FILTER, open(os.path.join(OUT_DIR, "fa_filter.json"), "w"), indent=1)


table = []
for s in SEVS:
    test_rows = collect(test_pairs, s, "ours")
    plain_rows = collect(test_pairs, s, "plain")
    for name, rows, thr, mpx in [("Detector alone", plain_rows, 0.5, 0),
                                 ("Ours (RG-DB joint)", test_rows, 0.5, 0),
                                 ("Ours + template check", test_rows, FILTER["score_thr"], FILTER["min_px"])]:
        P, R, F1 = prf(rows, thr, mpx)
        table.append({"severity": s, "method": name, "precision": P, "recall": R, "F1": F1,
                      "mAP@0.5": filtered_map(rows, mpx)})
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in table[-1].items()})
json.dump(table, open(os.path.join(OUT_DIR, "fa_filter_results.json"), "w"), indent=1)

print(f"\n{'':24s}" + "".join(f"   level {s} P / R / F1  " for s in SEVS))
for name in ["Detector alone", "Ours (RG-DB joint)", "Ours + template check"]:
    rows = {t["severity"]: t for t in table if t["method"] == name}
    print(f"{name:24s}" + "".join(f"   {rows[s]['precision']:.3f}/{rows[s]['recall']:.3f}/{rows[s]['F1']:.3f}" for s in SEVS))
