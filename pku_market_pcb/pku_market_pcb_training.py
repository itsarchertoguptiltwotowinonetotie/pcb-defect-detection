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


PKU_DIR = os.path.join(DATA_TMP, "PCB-DATASET")
if not os.path.exists(os.path.join(PKU_DIR, "PCB_USED")):
    print("downloading PKU-Market-PCB (about 3.7 GB, a few minutes)...")
    os.system(f"git clone --depth 1 https://github.com/Ironbrotherstyle/PCB-DATASET.git '{PKU_DIR}'")
PKU_OUT = os.path.join(PROJECT_DIR, "runs_pku")
os.makedirs(PKU_OUT, exist_ok=True)

PK = dict(
    scale=0.5,
    tile=640, overlap=160,
    split="random_811",
    init="deeppcb",
    use_template=True, use_nnvr=True,
    epochs=12, lr=0.005, lr_steps=(8, 11), weight_decay=1e-4, batch_size=2,
    crops_per_img=2,
    neg_prob=0.1,
    train_blur_prob=0.0,
    eval_severities=[0, 3, 5],
    run_zero_shot=True,
    num_workers=CFG["num_workers"], seed=CFG["seed"],
)
if SMOKE:
    PK.update(epochs=1, eval_severities=[0, 3], init="imagenet")
DETECTOR_CKPT = os.path.join(PROJECT_DIR, "runs_detector", "ref_nnvr_official_best.pth")
PKU_CLASS = {"open_circuit": 1, "short": 2, "mouse_bite": 3, "spur": 4, "spurious_copper": 5, "missing_hole": 6}
PKU_TAG = f"pku_{PK['split']}_{PK['init']}" + ("" if PK["use_template"] else "_testonly")
print("PKU:", PKU_DIR, "| outputs:", PKU_OUT, "| run:", PKU_TAG)


import glob, xml.etree.ElementTree as ET

def read_pku():
    items = []
    for xp in sorted(glob.glob(os.path.join(PKU_DIR, "Annotations", "*", "*.xml"))):
        r = ET.parse(xp).getroot()
        folder, name = os.path.basename(os.path.dirname(xp)), r.find("filename").text.strip()
        img = os.path.join(PKU_DIR, "images", folder, os.path.splitext(os.path.basename(xp))[0] + ".jpg")
        if not os.path.exists(img):
            img = os.path.join(PKU_DIR, "images", folder, name)
        board = os.path.basename(img)[:2]
        boxes, labels = [], []
        for o in r.findall("object"):
            b = o.find("bndbox")
            boxes.append([float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
            labels.append(PKU_CLASS[o.find("name").text.strip().lower()])
        items.append({"img": img, "temp": os.path.join(PKU_DIR, "PCB_USED", board + ".JPG"), "board": board,
                      "boxes": np.array(boxes, np.float32).reshape(-1, 4), "labels": np.array(labels, np.int64)})
    return items

def pku_splits(items, mode, seed):
    rng = random.Random(seed)
    if mode == "random_811":
        items = items[:]; rng.shuffle(items)
        n = len(items); a, b = int(0.8 * n), int(0.9 * n)
        return items[:a], items[a:b], items[b:]
    if mode == "board_out":
        test = [it for it in items if it["board"] in ("11", "12")]
        rest = [it for it in items if it["board"] not in ("11", "12")]; rng.shuffle(rest)
        n_val = len(rest) // 9
        return rest[n_val:], rest[:n_val], test
    raise ValueError(mode)

pku_items = read_pku()
assert len(pku_items) == 693, f"expected 693 photos, found {len(pku_items)}; is PKU_DIR right?"
pku_train, pku_val, pku_test = pku_splits(pku_items, PK["split"], PK["seed"])
if SMOKE:
    pku_train, pku_val, pku_test = pku_train[:6], pku_val[:2], pku_test[:2]
print(f"PKU {PK['split']}: train {len(pku_train)} | val {len(pku_val)} | test {len(pku_test)} photos, "
      f"{sum(len(it['labels']) for it in pku_items)} defects")


CACHE_DIR = os.path.join(DATA_TMP, f"pku_cache_s{PK['scale']}")
os.makedirs(CACHE_DIR, exist_ok=True)

def cached(path):
    out = os.path.join(CACHE_DIR, os.path.basename(os.path.dirname(path)) + "_" +
                       os.path.splitext(os.path.basename(path))[0] + ".png")
    if not os.path.exists(out):
        im = cv2.imread(path)
        im = cv2.resize(im, None, fx=PK["scale"], fy=PK["scale"], interpolation=cv2.INTER_AREA)
        cv2.imwrite(out, im, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return out

def load_rgb(path):
    return cv2.cvtColor(cv2.imread(cached(path)), cv2.COLOR_BGR2RGB)

t0 = time.time()
for it in pku_train + pku_val + pku_test:
    cached(it["img"]); cached(it["temp"])
print(f"cache ready in {time.time() - t0:.0f}s:", CACHE_DIR)


class PKUTiles(Dataset):
    def __init__(self, items, n_per_img=2):
        self.items, self.n = items, n_per_img

    def __len__(self):
        return len(self.items) * self.n

    def __getitem__(self, i):
        it, rng, T = self.items[i // self.n], np.random.default_rng(), PK["tile"]
        test, temp = load_rgb(it["img"]), load_rgb(it["temp"])
        H, W = test.shape[:2]
        if temp.shape[:2] != (H, W):
            temp = cv2.resize(temp, (W, H), interpolation=cv2.INTER_AREA)
        boxes, labels = it["boxes"] * PK["scale"], it["labels"]
        if len(boxes) and rng.random() >= PK["neg_prob"]:
            x1, y1, x2, y2 = boxes[rng.integers(len(boxes))]
            lo, hi = max(0, int(x2) - T), max(0, min(int(x1), W - T))
            cx = int(rng.integers(min(lo, hi), hi + 1))
            lo, hi = max(0, int(y2) - T), max(0, min(int(y1), H - T))
            cy = int(rng.integers(min(lo, hi), hi + 1))
        else:
            cx, cy = int(rng.integers(0, max(1, W - T + 1))), int(rng.integers(0, max(1, H - T + 1)))
        cx, cy = min(max(cx, 0), max(W - T, 0)), min(max(cy, 0), max(H - T, 0))
        test, temp = test[cy:cy + T, cx:cx + T], temp[cy:cy + T, cx:cx + T]
        b = boxes - np.array([cx, cy, cx, cy], np.float32)
        h, w = test.shape[:2]
        c = np.clip(b, 0, [w, h, w, h])
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        keep = ((c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1]) >= 0.5 * area) & (c[:, 2] - c[:, 0] >= 2) & (c[:, 3] - c[:, 1] >= 2)
        b, l = c[keep], labels[keep]
        if rng.random() < PK["train_blur_prob"]:
            test = degrade(test, int(rng.integers(1, 6)), rng)
        if rng.random() < 0.5:
            test, temp = test[:, ::-1], temp[:, ::-1]; b[:, [0, 2]] = w - b[:, [2, 0]]
        if rng.random() < 0.5:
            test, temp = test[::-1], temp[::-1]; b[:, [1, 3]] = h - b[:, [3, 1]]
        x = np.concatenate([test, temp], axis=2).astype(np.float32) / 255.0
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)
        return x, {"boxes": torch.from_numpy(np.ascontiguousarray(b, np.float32)).reshape(-1, 4),
                   "labels": torch.from_numpy(np.ascontiguousarray(l)), "image_id": torch.tensor(i)}

x0, t0_ = PKUTiles(pku_train)[0]
print("training tile:", tuple(x0.shape), "boxes", t0_["boxes"].shape[0], "labels", t0_["labels"].tolist())


from torchvision.ops import batched_nms

def tile_starts(n, T, step):
    if n <= T: return [0]
    s = list(range(0, n - T, step)); s.append(n - T)
    return s

@torch.no_grad()
def detect_photo(model, test, temp, bs=4, edge=4):
    T, step = PK["tile"], PK["tile"] - PK["overlap"]
    H, W = test.shape[:2]
    if H < T or W < T:
        ph, pw = max(T - H, 0), max(T - W, 0)
        test = np.pad(test, ((0, ph), (0, pw), (0, 0))); temp = np.pad(temp, ((0, ph), (0, pw), (0, 0)))
    x = torch.from_numpy(np.concatenate([test, temp], 2)).permute(2, 0, 1)
    Hp, Wp = test.shape[:2]
    jobs = [(y, x0) for y in tile_starts(Hp, T, step) for x0 in tile_starts(Wp, T, step)]
    allb, alls, alll = [], [], []
    for k in range(0, len(jobs), bs):
        chunk = jobs[k:k + bs]
        xs = [(x[:, y:y + T, x0:x0 + T].float() / 255.0).to(device) for y, x0 in chunk]
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outs = model(xs)
        for (y, x0), o in zip(chunk, outs):
            b = o["boxes"].float().cpu()
            ok = torch.ones(len(b), dtype=torch.bool)
            if x0 > 0: ok &= b[:, 0] > edge
            if y > 0: ok &= b[:, 1] > edge
            if x0 + T < Wp: ok &= b[:, 2] < T - edge
            if y + T < Hp: ok &= b[:, 3] < T - edge
            allb.append(b[ok] + torch.tensor([x0, y, x0, y], dtype=torch.float32))
            alls.append(o["scores"].float().cpu()[ok]); alll.append(o["labels"].cpu()[ok])
    b, s, l = torch.cat(allb), torch.cat(alls), torch.cat(alll)
    keep = batched_nms(b, s, l, 0.5)[:300]
    return {"boxes": b[keep], "scores": s[keep], "labels": l[keep]}, len(jobs)

def predict_pku(model, items, severity=0):
    model.eval()
    preds, gts, t, n_tiles = [], [], 0.0, 0
    for i, it in enumerate(items):
        test, temp = load_rgb(it["img"]), load_rgb(it["temp"])
        if temp.shape[:2] != test.shape[:2]:
            temp = cv2.resize(temp, (test.shape[1], test.shape[0]), interpolation=cv2.INTER_AREA)
        if severity > 0:
            test = degrade(test, severity, np.random.default_rng(PK["seed"] * 100003 + i))
        if not PK["use_template"]:
            temp = np.zeros_like(temp)
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        p, nt = detect_photo(model, test, temp)
        if device.type == "cuda": torch.cuda.synchronize()
        t += time.time() - t0; n_tiles += nt
        preds.append(p)
        gts.append({"boxes": torch.from_numpy(it["boxes"] * PK["scale"]), "labels": torch.from_numpy(it["labels"])})
    return preds, gts, len(items) / max(t, 1e-9)


def load_detector(path):
    st = torch.load(path, map_location=device, weights_only=False)
    m = build_model({**st["cfg"], "pretrained": False}).to(device)
    m.load_state_dict(st["model"]); return m, st["cfg"]

zs_path = os.path.join(PKU_OUT, f"pku_zeroshot_{PK['split']}.json")
if PK["run_zero_shot"] and os.path.exists(DETECTOR_CKPT):
    if os.path.exists(zs_path):
        zs = json.load(open(zs_path)); print("loaded:", zs_path)
    else:
        m0, _ = load_detector(DETECTOR_CKPT); m0.roi_heads.score_thresh = 0.01
        p, g, fps = predict_pku(m0, pku_test)
        zs = full_report(p, g, fps); json.dump(zs, open(zs_path, "w"), indent=1)
        del m0
    print(f"zero-shot (DeepPCB-trained, no PKU training): mAP@0.5 = {zs['mAP@0.5']:.4f}")
else:
    print("zero-shot skipped (needs runs_detector/ref_nnvr_official_best.pth in the pcb-checkpoints dataset)")


def pku_train_run():
    best_path, last_path = os.path.join(PKU_OUT, PKU_TAG + "_best.pth"), os.path.join(PKU_OUT, PKU_TAG + "_last.pth")
    hist_path = os.path.join(PKU_OUT, PKU_TAG + "_history.json")
    if os.path.exists(best_path) and os.path.exists(hist_path):
        print("already trained:", best_path); return best_path, json.load(open(hist_path))
    seed_everything(PK["seed"])
    mcfg = {**CFG, "use_template": PK["use_template"], "use_nnvr": PK["use_nnvr"],
            "pretrained": PK["init"] == "imagenet" and not SMOKE}
    model = build_model(mcfg).to(device)
    if PK["init"] == "deeppcb":
        assert os.path.exists(DETECTOR_CKPT), "init='deeppcb' needs runs_detector/ref_nnvr_official_best.pth (or set init='imagenet')"
        model.load_state_dict(torch.load(DETECTOR_CKPT, map_location=device, weights_only=False)["model"])
        print("started from the DeepPCB detector:", DETECTOR_CKPT)
    loader = DataLoader(PKUTiles(pku_train, PK["crops_per_img"]), batch_size=PK["batch_size"], shuffle=True,
                        num_workers=PK["num_workers"], collate_fn=collate, pin_memory=device.type == "cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=PK["lr"], momentum=0.9, weight_decay=PK["weight_decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(PK["lr_steps"]), gamma=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    warmup = min(300, len(loader))
    history, best, start, it = [], -1.0, 0, 0
    if os.path.exists(last_path):
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); history, best, start = st["history"], st["best"], st["epoch"]
        it = start * len(loader)
        print(f"resuming after epoch {start} (best val mAP so far {best:.4f})")

    for epoch in range(start, PK["epochs"]):
        model.train(); sums, n, t0 = {}, 0, time.time()
        for xs, ts in loader:
            xs = [x.to(device) for x in xs]
            ts = [{k: v.to(device) for k, v in t.items()} for t in ts]
            if it < warmup:
                for g in opt.param_groups: g["lr"] = PK["lr"] * (0.001 + 0.999 * it / warmup)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                losses = model(xs, ts); loss = sum(losses.values())
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch"); opt.zero_grad(); it += 1; continue
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 10.0); scaler.step(opt); scaler.update()
            for k, v in losses.items(): sums[k] = sums.get(k, 0.0) + v.item()
            n += 1; it += 1
        sched.step()
        model.roi_heads.score_thresh = 0.01
        p, g, _ = predict_pku(model, pku_val)
        model.roi_heads.score_thresh = 0.05
        val = evaluate_detections(p, g, 0.5)
        row = {"epoch": epoch + 1, **{k: v / max(n, 1) for k, v in sums.items()},
               "val_mAP@0.5": val["mAP"], "time_s": round(time.time() - t0, 1)}
        history.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        if val["mAP"] > best:
            best = val["mAP"]
            torch.save({"model": model.state_dict(), "cfg": mcfg, "pk": PK, "epoch": epoch + 1, "val_mAP@0.5": best}, best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "best": best, "history": history}, last_path)
    json.dump(history, open(hist_path, "w"), indent=1)
    print(f"best val mAP@0.5 = {best:.4f} -> {best_path}")
    return best_path, history

pku_best, pku_hist = pku_train_run()


res_path = os.path.join(PKU_OUT, PKU_TAG + "_results.json")
results = json.load(open(res_path)) if os.path.exists(res_path) else {}
model, _ = load_detector(pku_best); model.roi_heads.score_thresh = 0.01
for s in PK["eval_severities"]:
    if str(s) in results: continue
    p, g, fps = predict_pku(model, pku_test, severity=s)
    results[str(s)] = full_report(p, g, fps)
    json.dump(results, open(res_path, "w"), indent=1)
if "0" in results and "params_M" not in results["0"]:
    results["0"]["params_M"] = sum(q.numel() for q in model.parameters()) / 1e6
    json.dump(results, open(res_path, "w"), indent=1)

r0 = results["0"]
print(f"PKU test ({len(pku_test)} photos), run {PKU_TAG}")
print(f"  mAP@0.5 {r0['mAP@0.5']:.4f} | mAP@0.5:0.95 {r0['mAP@0.5:0.95']:.4f} | P {r0['P@0.5']:.3f} "
      f"R {r0['R@0.5']:.3f} F1 {r0['F1@0.5']:.3f} | {r0['FPS']:.2f} photos/s")
print("  per class AP@0.5:", {k: round(v, 3) for k, v in r0["AP@0.5 per class"].items()})
print("  blur levels:", {f"level {s}": round(results[str(s)]["mAP@0.5"], 4) for s in PK["eval_severities"] if str(s) in results})
print("\nComparison (mAP@0.5 on PKU-Market-PCB, 8:1:1):")
print(f"  MGRE-Net (base paper)        : 0.964")
if PK["run_zero_shot"] and os.path.exists(zs_path):
    print(f"  Ours, DeepPCB model, no PKU training: {json.load(open(zs_path))['mAP@0.5']:.4f}")
print(f"  Ours, fine-tuned on PKU      : {r0['mAP@0.5']:.4f}")


import matplotlib
if not IN_COLAB and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

def show_pku(idx=0, severity=0, thr=0.5, save=None):
    it = pku_test[idx]
    test, temp = load_rgb(it["img"]), load_rgb(it["temp"])
    if temp.shape[:2] != test.shape[:2]:
        temp = cv2.resize(temp, (test.shape[1], test.shape[0]), interpolation=cv2.INTER_AREA)
    if severity > 0:
        test = degrade(test, severity, np.random.default_rng(PK["seed"] * 100003 + idx))
    p, _ = detect_photo(model, test, temp if PK["use_template"] else np.zeros_like(temp))
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].imshow(temp); ax[0].set_title("template (defect-free board)"); ax[0].axis("off")
    ax[1].imshow(test); ax[1].axis("off")
    for x1, y1, x2, y2 in (it["boxes"] * PK["scale"]).tolist():
        ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=2))
    for b, l, s in zip(p["boxes"], p["labels"], p["scores"]):
        if s < thr: continue
        x1, y1, x2, y2 = b.tolist()
        ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
        ax[1].text(x1, y1 - 4, f"{CLASS_NAMES[int(l)]} {s:.2f}", color="white", fontsize=8, bbox=dict(fc="red", alpha=0.7, pad=1))
    ax[1].set_title(f"{os.path.basename(it['img'])}, blur level {severity}: green = true, red = ours")
    plt.tight_layout()
    if save: plt.savefig(save, dpi=90)
    plt.show()

show_pku(0, 0, save=os.path.join(PKU_OUT, "pku_example.png"))
