import os, gc, json, zlib, threading, urllib.request
from collections import OrderedDict
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, box_iou, batched_nms

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("PCB_WEIGHTS", os.path.join(ROOT, "weights"))
CLASS_NAMES = ["background", "open", "short", "mousebite", "spur", "spurious copper", "pin-hole"]
NUM_CLASSES = len(CLASS_NAMES)

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


BLUR_TYPES = ["None", "Motion blur", "Out-of-focus blur", "Gaussian blur (unseen)",
              "Stronger than level 5 (unseen)", "Low resolution (unseen)"]


def apply_blur(img, kind, level, seed_key):
    if kind == "None" or (level <= 0 and "unseen" not in kind):
        return img
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()) + 1000 * int(level))
    x = img.astype(np.float32)
    if kind == "Motion blur":
        x, noise = cv2.filter2D(x, -1, motion_kernel(MOTION_LEN[level], rng.uniform(0, 180))), NOISE_STD[level]
    elif kind == "Out-of-focus blur":
        x, noise = cv2.filter2D(x, -1, defocus_kernel(DEFOCUS_R[level])), NOISE_STD[level]
    elif kind.startswith("Gaussian"):
        x, noise = cv2.GaussianBlur(x, (0, 0), 3.0), 4
    elif kind.startswith("Stronger"):
        k = motion_kernel(29, rng.uniform(0, 180)) if rng.random() < 0.5 else defocus_kernel(7)
        x, noise = cv2.filter2D(x, -1, k), 7
    else:
        h, w = x.shape[:2]
        x = cv2.resize(cv2.resize(x, (w // 3, h // 3), interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_CUBIC)
        noise = 3
    x += rng.normal(0, noise, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


class RefGuidedBackbone(nn.Module):
    def __init__(self, pretrained=False, use_template=True, trainable_layers=3):
        super().__init__()
        self.body = resnet_fpn_backbone(backbone_name="resnet50", weights=None, trainable_layers=trainable_layers)
        self.out_channels = 256
        self.use_template = use_template
        if use_template:
            self.fuse = nn.ModuleDict({
                k: nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 1))
                for k in ["0", "1", "2", "3", "pool"]})

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


def build_detector(cfg):
    backbone = RefGuidedBackbone(False, cfg.get("use_template", True))
    anchors = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    pooler = CountingRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    head = NNVRBoxHead(pooler, use_nnvr=cfg.get("use_nnvr", True))
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return FasterRCNN(backbone, rpn_anchor_generator=anchors, box_roi_pool=pooler, box_head=head,
                      box_predictor=FastRCNNPredictor(1024, NUM_CLASSES), min_size=640, max_size=640,
                      image_mean=mean * 2, image_std=std * 2, box_detections_per_img=100)


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


MODEL_FILES = {
    "det_main": "det_main.pt",
    "det_testonly_nnvr": "det_testonly_nnvr.pt",
    "det_plain": "det_plain.pt",
    "det_blur": "det_blur.pt",
    "final": "final.pt",
    "rg_template": "rg_template.pt",
    "rg_notemplate": "rg_notemplate.pt",
    "det_pku": "det_pku.pt",
}
_CACHE = OrderedDict()
MAX_LOADED = int(os.environ.get("PCB_MAX_LOADED", 4))
URL_FILE = os.path.join(ROOT, "weights_url.txt")


def ensure_weights(log=print):
    if not os.path.exists(URL_FILE):
        return
    base = open(URL_FILE).read().strip().rstrip("/")
    os.makedirs(WEIGHTS, exist_ok=True)
    for name, f in MODEL_FILES.items():
        path = os.path.join(WEIGHTS, f)
        if os.path.exists(path):
            continue
        log(f"Downloading {f} ...")
        try:
            tmp = path + ".part"
            urllib.request.urlretrieve(f"{base}/{f}", tmp)
            os.replace(tmp, path)
        except Exception as e:
            log(f"Could not download {f}: {e}")


def available(name):
    return os.path.exists(os.path.join(WEIGHTS, MODEL_FILES[name]))


def _f32(sd):
    return {k: (v.float() if torch.is_floating_point(v) else v) for k, v in sd.items()}


def _det_from(sd, cfg):
    m = build_detector(cfg)
    m.load_state_dict(_f32(sd))
    m.roi_heads.score_thresh = 0.01
    return m.eval().to(DEVICE)


def _rg_from(sd, in_template):
    m = RGDBNet(in_template, 32)
    m.load_state_dict(_f32(sd))
    return m.eval().to(DEVICE)


_LOCK = threading.Lock()


def get(name):
    with _LOCK:
        return _get(name)


def _get(name):
    if name in _CACHE:
        _CACHE.move_to_end(name)
        return _CACHE[name]
    while len(_CACHE) >= MAX_LOADED:
        _CACHE.popitem(last=False)
        gc.collect()
    st = torch.load(os.path.join(WEIGHTS, MODEL_FILES[name]), map_location="cpu", weights_only=True)
    if name == "final":
        obj = (_det_from(st["det"], st["cfg"]), _rg_from(st["rg"], True))
    elif name.startswith("rg_"):
        obj = _rg_from(st["state"], name == "rg_template")
    else:
        obj = _det_from(st["state"], st["cfg"])
    _CACHE[name] = obj
    return obj


def to_t3(img):
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def to_t1(img):
    return to_t3(img).mean(0, keepdim=True)[None]


@torch.no_grad()
def detect(model, test_rgb, temp_rgb):
    t = test_rgb if torch.is_tensor(test_rgb) else to_t3(test_rgb)
    r = temp_rgb if torch.is_tensor(temp_rgb) else to_t3(temp_rgb)
    o = model([torch.cat([t, r], 0).to(DEVICE)])[0]
    return {k: v.cpu().numpy() for k, v in o.items()}


@torch.no_grad()
def restore(rg, blur_rgb, temp_rgb):
    h, w = blur_rgb.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    b, r = to_t1(blur_rgb), to_t1(temp_rgb)
    if ph or pw:
        b, r = F.pad(b, (0, pw, 0, ph), mode="replicate"), F.pad(r, (0, pw, 0, ph), mode="replicate")
    p = torch.sigmoid(rg(b.to(DEVICE), r.to(DEVICE)))[0, 0, :h, :w]
    return p.cpu().numpy()


def otsu(blur_rgb):
    g = cv2.cvtColor(blur_rgb, cv2.COLOR_RGB2GRAY)
    _, b = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b.astype(np.float32) / 255.0


def gray3(p):
    g = (np.clip(p, 0, 1) * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, 2)


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_CONSTANT)
    mu_a, mu_b = g(a), g(b)
    va, vb, cov = g(a * a) - mu_a ** 2, g(b * b) - mu_b ** 2, g(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * cov + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


def keep(out, thr):
    m = out["scores"] >= thr
    return {k: v[m] for k, v in out.items()}


def match(out, gt_boxes, gt_labels):
    pb, pl = torch.as_tensor(out["boxes"]), torch.as_tensor(out["labels"])
    gb, gl = torch.as_tensor(np.asarray(gt_boxes, np.float32).reshape(-1, 4)), torch.as_tensor(np.asarray(gt_labels))
    used, ok = set(), []
    order = np.argsort(-out["scores"]) if len(out["scores"]) else []
    flags = [False] * len(pb)
    for i in order:
        if len(gb):
            ious = box_iou(pb[i][None], gb)[0].clone()
            ious[gl != pl[i]] = 0
            for j in used:
                ious[j] = 0
            j = int(ious.argmax())
            if ious[j] >= 0.5:
                used.add(j); flags[i] = True
    tp = sum(flags)
    return {"found": tp, "missed": len(gb) - tp, "false alarms": len(pb) - tp, "total": len(gb),
            "flags": flags, "missed_idx": [j for j in range(len(gb)) if j not in used]}


def tile_starts(n, T, step):
    if n <= T:
        return [0]
    s = list(range(0, n - T, step)); s.append(n - T)
    return s


@torch.no_grad()
def detect_photo(model, test, temp, T=640, overlap=160, edge=4):
    step = T - overlap
    H, W = test.shape[:2]
    if H < T or W < T:
        ph, pw = max(T - H, 0), max(T - W, 0)
        test = np.pad(test, ((0, ph), (0, pw), (0, 0))); temp = np.pad(temp, ((0, ph), (0, pw), (0, 0)))
    Hp, Wp = test.shape[:2]
    allb, alls, alll = [], [], []
    for y in tile_starts(Hp, T, step):
        for x0 in tile_starts(Wp, T, step):
            o = detect(model, test[y:y + T, x0:x0 + T], temp[y:y + T, x0:x0 + T])
            b = torch.from_numpy(o["boxes"]).float()
            ok = torch.ones(len(b), dtype=torch.bool)
            if x0 > 0: ok &= b[:, 0] > edge
            if y > 0: ok &= b[:, 1] > edge
            if x0 + T < Wp: ok &= b[:, 2] < T - edge
            if y + T < Hp: ok &= b[:, 3] < T - edge
            allb.append(b[ok] + torch.tensor([x0, y, x0, y], dtype=torch.float32))
            alls.append(torch.from_numpy(o["scores"])[ok]); alll.append(torch.from_numpy(o["labels"])[ok])
    b, s, l = torch.cat(allb), torch.cat(alls), torch.cat(alll)
    k = batched_nms(b, s, l, 0.5)[:300]
    return {"boxes": b[k].numpy(), "scores": s[k].numpy(), "labels": l[k].numpy()}


CACHE_FILE = os.path.join(ROOT, "cache", "precomputed.json")
_DET = {}
if os.path.exists(CACHE_FILE):
    _DET = {k: {kk: np.asarray(vv, np.float32 if kk != "labels" else np.int64) for kk, vv in v.items()}
            for k, v in json.load(open(CACHE_FILE)).items()}
RECORD = False


def cached(key, fn):
    if key in _DET:
        return _DET[key]
    out = fn()
    out = {"boxes": np.asarray(out["boxes"], np.float32).reshape(-1, 4), "scores": np.asarray(out["scores"], np.float32),
           "labels": np.asarray(out["labels"], np.int64)}
    _DET[key] = out
    return out


def save_cache():
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    json.dump({k: {kk: np.round(vv, 4).tolist() for kk, vv in v.items()} for k, v in _DET.items()}, open(CACHE_FILE, "w"))
