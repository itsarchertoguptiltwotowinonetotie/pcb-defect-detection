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

CFG.update(use_template=False, use_nnvr=False, epochs=14, lr_steps=(9, 12))
if SMOKE:
    CFG.update(epochs=1)

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

model = build_model(CFG).to(device)
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")


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


def run_name(cfg):
    return f"{'ref' if cfg['use_template'] else 'testonly'}_{'nnvr' if cfg['use_nnvr'] else 'plain'}_{cfg['split']}" \
           + (f"_blur{cfg['train_blur_prob']}" if cfg["train_blur_prob"] > 0 else "")

def train(model, cfg):
    train_loader = make_loader(train_pairs, train=True, shuffle=True)
    val_loader = make_loader(val_pairs)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(cfg["lr_steps"]), gamma=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    warmup_iters = min(500, len(train_loader))
    ckpt = os.path.join(OUT_DIR, run_name(cfg) + "_best.pth")
    last = os.path.join(OUT_DIR, run_name(cfg) + "_last.pth")
    hist_path = os.path.join(OUT_DIR, run_name(cfg) + "_history.json")
    if os.path.exists(ckpt) and os.path.exists(hist_path):
        print("already trained:", ckpt); return ckpt, json.load(open(hist_path))
    history, best, it, start = [], -1.0, 0, 0
    if os.path.exists(last):
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); history, best, start = st["history"], st["best"], st["epoch"]
        it = start * len(train_loader)
        print(f"resuming after epoch {start} (best val mAP so far {best:.4f})")

    for epoch in range(start, cfg["epochs"]):
        model.train()
        sums, n = {}, 0
        t0 = time.time()
        for xs, ts in train_loader:
            xs = [x.to(device) for x in xs]
            ts = [{k: v.to(device) for k, v in t.items()} for t in ts]
            if it < warmup_iters:
                for g in opt.param_groups:
                    g["lr"] = cfg["lr"] * (0.001 + 0.999 * it / warmup_iters)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                losses = model(xs, ts)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                print("non-finite loss, skipping batch"); opt.zero_grad(); it += 1; continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
            n += 1; it += 1
        sched.step()
        preds, gts, _ = predict(model, val_loader)
        val = evaluate_detections(preds, gts, 0.5)
        row = {"epoch": epoch + 1, **{k: v / max(n, 1) for k, v in sums.items()},
               "val_mAP@0.5": val["mAP"], "time_s": round(time.time() - t0, 1)}
        history.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        if val["mAP"] > best:
            best = val["mAP"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch + 1, "val_mAP@0.5": best}, ckpt)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "best": best, "history": history}, last)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=1)
    print(f"best val mAP@0.5 = {best:.4f} -> {ckpt}")
    return ckpt, history

ckpt_path, history = train(model, CFG)


ckpt_path = os.path.join(OUT_DIR, run_name(CFG) + "_best.pth")
state = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(state["model"])
model.roi_heads.score_thresh = 0.01
test_loader = make_loader(test_pairs)
preds, gts, fps = predict(model, test_loader)
report = full_report(preds, gts, fps)
report["params_M"] = sum(p.numel() for p in model.parameters()) / 1e6
try:
    report["GFLOPs"] = count_gflops(model)
except Exception as e:
    print("GFLOPs count skipped:", e)
print(json.dumps(report, indent=1))
with open(os.path.join(OUT_DIR, run_name(CFG) + "_test_clean.json"), "w") as f:
    json.dump(report, f, indent=1)


severities = [0, 1, 2, 3, 4, 5] if not SMOKE else [0, 3]
blur_path = os.path.join(OUT_DIR, run_name(CFG) + "_test_blur.json")
blur_rows = json.load(open(blur_path)) if os.path.exists(blur_path) else []
for s in severities:
    if any(r["severity"] == s for r in blur_rows):
        continue
    p, g, _ = predict(model, make_loader(test_pairs, severity=s))
    r = full_report(p, g)
    blur_rows.append({"severity": s, "mAP@0.5": r["mAP@0.5"], "F1@0.5": r["F1@0.5"]})
    print(blur_rows[-1])
    with open(blur_path, "w") as f:
        json.dump(blur_rows, f, indent=1)
print({r["severity"]: round(r["mAP@0.5"], 4) for r in blur_rows})


import matplotlib
if not IN_COLAB and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

def show(idx=0, severity=0, score_thr=0.5, save=None):
    ds = DeepPCBPairs(test_pairs, eval_severity=severity, seed=CFG["seed"])
    x, t = ds[idx]
    model.eval()
    with torch.no_grad():
        o = model([x.to(device)])[0]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    ax[0].imshow(x[3:].permute(1, 2, 0).numpy()); ax[0].set_title("Template (defect-free)")
    ax[1].imshow(x[:3].permute(1, 2, 0).numpy()); ax[1].set_title(f"Test image, blur severity {severity}")
    for b in t["boxes"]:
        x1, y1, x2, y2 = b.tolist()
        ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=2))
    for b, l, s in zip(o["boxes"].cpu(), o["labels"].cpu(), o["scores"].cpu()):
        if s < score_thr:
            continue
        x1, y1, x2, y2 = b.tolist()
        ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="red", lw=1.5))
        ax[1].text(x1, y1 - 3, f"{CLASS_NAMES[int(l)]} {s:.2f}", color="white", fontsize=8,
                   bbox=dict(fc="red", alpha=0.7, pad=1))
    for a in ax: a.axis("off")
    plt.tight_layout()
    if save: plt.savefig(save, dpi=120)
    plt.show()

show(0, severity=0, save=os.path.join(OUT_DIR, run_name(CFG) + "_example_clean.png"))
show(0, severity=3, save=os.path.join(OUT_DIR, run_name(CFG) + "_example_blur3.png"))


import zipfile
def export_official(preds, pairs, path):
    with zipfile.ZipFile(path, "w") as z:
        for p, (tp, _, _) in zip(preds, pairs):
            name = os.path.basename(tp).replace("_test.jpg", ".txt")
            lines = [f"{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])},{float(s):.4f},{CLASS_NAMES[int(l)]}"
                     for b, s, l in zip(p["boxes"], p["scores"], p["labels"])]
            z.writestr(name, "\n".join(lines))
    print("wrote", path)

if CFG["split"] == "official" and not SMOKE:
    export_official(preds, test_pairs, os.path.join(OUT_DIR, run_name(CFG) + "_res.zip"))
