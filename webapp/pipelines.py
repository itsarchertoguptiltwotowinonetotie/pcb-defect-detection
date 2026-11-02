import os, io, json, time, base64, random, html
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import pcb_core as core

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")
DEEP = json.load(open(os.path.join(SAMPLES, "deeppcb", "index.json")))
PKU = json.load(open(os.path.join(SAMPLES, "pku", "index.json")))
NAMES = core.CLASS_NAMES

SRC_DEEP, SRC_PKU, SRC_UP = "DeepPCB test set", "PKU real photos", "Upload your own"
MODE_SYS, MODE_EXP = "Our system", "Experiments"
EXP_ABL = "What each part adds (ablation)"
EXP_BLUR = "Blur handling: 4 methods compared"
EXP_TPL = "Does the cleaner need the template?"
EXP_PKU = "Real photos: PKU dataset"
EXPS = [EXP_ABL, EXP_BLUR, EXP_TPL, EXP_PKU]

try:
    FONT = ImageFont.load_default(size=15)
except TypeError:
    FONT = ImageFont.load_default()


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def sample(source, idx):
    if source == SRC_PKU:
        it = PKU[idx]
        d = os.path.join(SAMPLES, "pku")
        return dict(id=it["id"], test=load_rgb(os.path.join(d, it["id"] + ".jpg")),
                    temp=load_rgb(os.path.join(d, it["template"] + ".jpg")), boxes=it["boxes"], labels=it["labels"])
    it = DEEP[idx]
    d = os.path.join(SAMPLES, "deeppcb")
    return dict(id=it["id"], test=load_rgb(os.path.join(d, it["id"] + "_test.jpg")),
                temp=load_rgb(os.path.join(d, it["id"] + "_temp.jpg")), boxes=it["boxes"], labels=it["labels"])


def gallery_items(source, defect):
    data = PKU if source == SRC_PKU else DEEP
    out, idxs = [], []
    for i, it in enumerate(data):
        names = sorted({NAMES[l] for l in it["labels"]})
        if defect != "All types" and defect not in names:
            continue
        path = os.path.join(SAMPLES, "pku", it["id"] + ".jpg") if source == SRC_PKU else \
            os.path.join(SAMPLES, "deeppcb", it["id"] + "_test.jpg")
        out.append(path)
        idxs.append(i)
    return out, idxs


COL_PRED, COL_GT = (220, 38, 38), (22, 163, 74)


def draw(img, out=None, gt=None, scale=1.0, width=2):
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    if gt is not None:
        for b in np.asarray(gt, np.float32).reshape(-1, 4):
            d.rectangle([v * scale for v in b], outline=COL_GT, width=width)
    if out is not None:
        for b, l, s in zip(out["boxes"], out["labels"], out["scores"]):
            x1, y1, x2, y2 = [v * scale for v in b]
            d.rectangle([x1, y1, x2, y2], outline=COL_PRED, width=width)
            t = f"{NAMES[int(l)]} {s:.2f}"
            tw = d.textlength(t, font=FONT)
            ty = y1 - 18 if y1 > 20 else y2 + 2
            d.rectangle([x1, ty, x1 + tw + 6, ty + 17], fill=COL_PRED)
            d.text((x1 + 3, ty), t, fill=(255, 255, 255), font=FONT)
    return np.asarray(im)


def diff_overlay(clean, temp_gray):
    d = ((clean > 0.5) != (temp_gray > 0.5)).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    out = np.full(clean.shape + (3,), 255, np.uint8)
    out[temp_gray <= 0.5] = (215, 219, 226)
    out[cv2.dilate(d, np.ones((3, 3), np.uint8)) > 0] = COL_PRED
    return out


def b64(img, max_w=900):
    im = Image.fromarray(img)
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fig(img, caption, sub=""):
    s = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
    return f'<figure><img src="{b64(img)}"/><figcaption><b>{html.escape(caption)}</b>{s}</figcaption></figure>'


def card(n, title, text, body):
    return (f'<section class="card"><div class="head"><span class="num">{n}</span><div><h3>{html.escape(title)}</h3>'
            f'<p>{text}</p></div></div>{body}</section>')


def grid(*figs):
    cols = 2 if len(figs) == 4 else min(len(figs), 3)
    return f'<div class="grid g{cols}">' + "".join(figs) + "</div>"


def chips(items):
    return '<div class="chips">' + "".join(
        f'<div class="chip {c}"><span>{html.escape(str(v))}</span><small>{html.escape(k)}</small></div>'
        for k, v, c in items) + "</div>"


def count_chips(m):
    p = m["found"] / max(m["found"] + m["false alarms"], 1)
    r = m["found"] / max(m["total"], 1)
    return chips([("defects found", f'{m["found"]} / {m["total"]}', "good"),
                  ("missed", m["missed"], "bad" if m["missed"] else ""),
                  ("false alarms", m["false alarms"], "bad" if m["false alarms"] else ""),
                  ("precision", f"{p:.0%}", ""), ("recall", f"{r:.0%}", "")])


def short_counts(m):
    return f'found {m["found"]}/{m["total"]} · missed {m["missed"]} · false alarms {m["false alarms"]}'


def table(rows, head):
    h = "".join(f"<th>{html.escape(x)}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tblwrap"><table class="tbl"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def det_table(out, flags=None):
    if len(out["scores"]) == 0:
        return '<p class="muted">No defects above the confidence threshold.</p>'
    rows = []
    for i, (b, l, s) in enumerate(zip(out["boxes"], out["labels"], out["scores"])):
        x1, y1, x2, y2 = [int(v) for v in b]
        ok = "" if flags is None else ('<span class="ok">correct</span>' if flags[i] else '<span class="no">false alarm</span>')
        rows.append([str(i + 1), NAMES[int(l)], f"{s:.2f}", f"({x1}, {y1}) to ({x2}, {y2})", ok])
    return table(rows, ["#", "defect type", "confidence", "box (x, y)", "check"])


def note(text):
    return f'<div class="note">{text}</div>'


def missing(names):
    return note("This part needs model files that are not uploaded yet: " + ", ".join(core.MODEL_FILES[n] for n in names) +
                ". Run the deploy notebook again with these files in your Drive.")


PLACEHOLDER = ('<div class="empty"><div class="big">Pick an image on the left and press <b>Run</b>.</div>'
               '<div class="muted">Results appear here step by step.</div></div>')


def key(model, s, kind, level, tag="raw"):
    lv = level if kind != "None" and "unseen" not in kind else 0
    return f"{model}|{tag}|{s['id']}|{kind}|{lv}"


def cdet(model, s, img, temp, kind, level, tag="raw"):
    run = lambda: core.detect(core.get(model) if model != "final" else core.get("final")[0], img, temp)
    return run() if s["id"] == "upload" else core.cached(key(model, s, kind, level, tag), run)


def cphoto(model, s, img, kind, level):
    if s["id"] == "upload":
        return core.detect_photo(core.get(model), img, s["temp"])
    return core.cached(key(model, s, kind, level, "photo"), lambda: core.detect_photo(core.get(model), img, s["temp"]))


def blur_text(kind, level):
    if kind == "None":
        return "no blur added (sharp image)"
    return kind if "unseen" in kind else f"{kind.lower()}, level {level}"


def run_system(s, kind, level, thr, known_gt):
    need = [n for n in ["final"] if not core.available(n)]
    if need:
        return missing(need)
    t0 = time.time()
    blur = core.apply_blur(s["test"], kind, level, s["id"])
    det, rg = core.get("final")
    clean = core.restore(rg, blur, s["temp"])
    temp_g = s["temp"].mean(2) / 255.0
    out = core.keep(cdet("final", s, core.gray3(clean), s["temp"], kind, level, "clean"), thr)
    dt = time.time() - t0
    parts = []
    parts.append(card(1, "Input pair", "The test image (with simulated camera blur if chosen) and its defect-free template.",
                      grid(fig(blur, "Test image", blur_text(kind, level)), fig(s["temp"], "Template (perfect board)"))))
    figs2 = [fig(core.gray3(clean), "Cleaned image (black and white)")]
    if known_gt:
        true = (s["test"].mean(2) > 127).astype(np.float32)
        figs2[0] = fig(core.gray3(clean), "Cleaned image (black and white)",
                       f"SSIM vs the true sharp image: blurry {core.ssim(blur.mean(2) / 255.0, true):.2f} → cleaned {core.ssim(clean, true):.2f}")
        figs2.append(fig(core.gray3(true), "True sharp image (for comparison)", "from the dataset; not used by the model"))
    parts.append(card(2, "Deblur + binarize", "A small U-Net turns the blurry photo into a clean black-and-white board. "
                      "It sees the template too, but is trained not to copy it, so defects stay.", grid(*figs2)))
    parts.append(card(3, "Compare with the template",
                      "Inside the detector, the features of the cleaned image are compared with the template's features. "
                      "This picture shows the same idea at pixel level: red = where the board differs from the perfect one.",
                      grid(fig(diff_overlay(clean, temp_g), "Difference from template"))))
    flags = None
    body = ""
    if known_gt:
        m = core.match(out, s["boxes"], s["labels"])
        flags = m["flags"]
    parts.append(card(4, "Detected defects",
                      f"Faster R-CNN with template comparison and NNVR. Boxes with confidence ≥ {thr:.2f} are kept.",
                      grid(fig(draw(core.gray3(clean), out), f"{len(out['scores'])} defects found", f"{dt:.1f} s on this server"))
                      + det_table(out, flags)))
    if known_gt:
        parts.append(card(5, "Check against the true answer",
                          "Green boxes are the real defects from the dataset labels. A detection counts as correct if it has "
                          "the right type and overlaps a real defect by at least 50 % (IoU ≥ 0.5).",
                          count_chips(m) + grid(fig(draw(core.gray3(clean), out, s["boxes"]), "Red = ours, green = true defects"))))
    return "".join(parts)


def run_pku_system(s, kind, level, thr, known=True):
    if not core.available("det_pku"):
        return missing(["det_pku"])
    t0 = time.time()
    test = core.apply_blur(s["test"], kind, level, s["id"])
    out = core.keep(cphoto("det_pku", s, test, kind, level), thr)
    dt = time.time() - t0
    m = core.match(out, s["boxes"], s["labels"]) if known else None
    what = "A real colour photo from PKU-Market-PCB" if known else "Your colour photo"
    parts = [card(1, "Input pair", f"{what} and the defect-free photo of the same board.",
                  grid(fig(test, "Test photo", blur_text(kind, level)), fig(s["temp"], "Template photo"))),
             card(2, "Detected defects", "The same detector, fine-tuned on PKU photos. The photo is checked in overlapping "
                  "640 × 640 tiles. The deblur stage is skipped here: it was trained on DeepPCB's black-and-white images.",
                  grid(fig(draw(test, out, width=3), f"{len(out['scores'])} defects found", f"{dt:.1f} s on this server"))
                  + det_table(out, m["flags"] if known else None))]
    if known:
        parts.append(card(3, "Check against the true answer", "Green = real defects, red = ours.",
                          count_chips(m) + grid(fig(draw(test, out, s["boxes"], width=3), "Red = ours, green = true defects"))))
    if kind != "None":
        parts.insert(0, note("Note: the PKU model was never trained on blur, so blurry colour photos are a known limitation "
                             "(mAP 42.9 at level 3, 8.9 at level 5)."))
    return "".join(parts)


def run_ablation(s, kind, level, thr):
    names = ["det_plain", "det_testonly_nnvr", "det_main"]
    need = [n for n in names if not core.available(n)]
    if need:
        return missing(need)
    test = core.apply_blur(s["test"], kind, level, s["id"])
    titles = ["Plain Faster R-CNN", "+ NNVR", "+ template comparison (ours)"]
    figs = []
    for n, t in zip(names, titles):
        out = core.keep(cdet(n, s, test, s["temp"], kind, level), thr)
        m = core.match(out, s["boxes"], s["labels"])
        figs.append(fig(draw(test, out, s["boxes"]), t, short_counts(m)))
    overall = table([["Plain Faster R-CNN", "97.0", "82.2", "89.3"], ["+ NNVR", "97.2", "88.5", "92.8"],
                     ["<b>+ template comparison (ours)</b>", "<b>99.4</b>", "<b>96.0</b>", "<b>97.4</b>"]],
                    ["model", "mAP@0.5 (%)", "precision (%)", "F1 (%)"])
    return (card(1, "Same image, three models", "Each model adds one part. Red = model output, green = true defects. "
                 f"Confidence threshold {thr:.2f}.", grid(*figs))
            + card(2, "On the whole DeepPCB test set (500 images)",
                   "Only one thing changes between rows, so the gain comes from that part. The template mostly removes false alarms.",
                   overall))


def run_blur(s, kind, level, thr):
    names = ["det_main", "det_blur", "final"]
    need = [n for n in names if not core.available(n)]
    if need:
        return missing(need)
    blur = core.apply_blur(s["test"], kind, level, s["id"])
    temp = s["temp"]
    rows = []
    o1 = core.keep(cdet("det_main", s, blur, temp, kind, level), thr)
    ots = core.gray3(core.otsu(blur))
    o2 = core.keep(cdet("det_main", s, ots, temp, kind, level, "otsu"), thr)
    o3 = core.keep(cdet("det_blur", s, blur, temp, kind, level), thr)
    clean = core.restore(core.get("final")[1], blur, temp)
    o4 = core.keep(cdet("final", s, core.gray3(clean), temp, kind, level, "clean"), thr)
    figs = []
    for img, out, t in [(blur, o1, "Trained on sharp images"), (ots, o2, "Classic Otsu binarize + detector"),
                        (blur, o3, "Trained on blurry images"), (core.gray3(clean), o4, "Ours: deblur + binarize + detector")]:
        figs.append(fig(draw(img, out, s["boxes"]), t, short_counts(core.match(out, s["boxes"], s["labels"]))))
    overall = table([["Trained on sharp images", "99.4", "94.9", "88.3", "76.4", "93.0", "58.9", "98.6"],
                     ["Otsu binarize + detector", "99.4", "86.5", "73.4", "57.5", "-", "-", "-"],
                     ["Trained on blurry images", "99.5", "99.1", "98.3", "96.3", "98.4", "<b>91.9</b>", "99.4"],
                     ["<b>Ours (deblur + binarize)</b>", "<b>99.5</b>", "<b>99.1</b>", "<b>98.7</b>", "<b>96.5</b>", "<b>98.9</b>",
                      "89.3", "<b>99.5</b>"]],
                    ["method", "sharp", "level 3", "level 4", "level 5", "Gaussian*", "stronger*", "low-res*"])
    return (card(1, "Same blurry image, four ways to handle blur", f"{blur_text(kind, level)}. Red = model output, "
                 f"green = true defects. Confidence threshold {thr:.2f}.", grid(*figs))
            + card(2, "On the whole DeepPCB test set (mAP@0.5, %)",
                   "* unseen blur types, never used in training. Honest reading: training on blur gives most of the "
                   "robustness; our stage keeps the same accuracy and adds a clean image a person can check "
                   "(SSIM 0.65 → 0.97 at level 5).", overall))


def run_template(s, kind, level, thr):
    names = ["rg_template", "rg_notemplate"]
    need = [n for n in names if not core.available(n)]
    if need:
        return missing(need)
    blur = core.apply_blur(s["test"], kind, level, s["id"])
    true = (s["test"].mean(2) > 127).astype(np.float32)
    a = core.restore(core.get("rg_template"), blur, s["temp"])
    b = core.restore(core.get("rg_notemplate"), blur, s["temp"])
    figs = [fig(blur, "Blurry input", f"SSIM {core.ssim(blur.mean(2) / 255.0, true):.3f}"),
            fig(core.gray3(a), "Cleaner with template", f"SSIM {core.ssim(a, true):.3f}"),
            fig(core.gray3(b), "Cleaner without template", f"SSIM {core.ssim(b, true):.3f}"),
            fig(core.gray3(true), "True sharp image")]
    overall = table([["with template", "0.970", "85.5", "78.7"], ["without template", "0.965", "83.6", "73.9"]],
                    ["deblur + binarize stage", "SSIM at level 5", "mAP@0.5 at level 5 (%)", "F1 at level 5 (%)"])
    return (card(1, "Same blurry image, cleaned with and without the template",
                 f"{blur_text(kind, level)}. SSIM: 1.0 means identical to the true sharp image.", grid(*figs))
            + card(2, "On the whole DeepPCB test set (stage A, before joint training)",
                   "The template barely changes image quality; it helps detection when blur is heavy.", overall))


def run_pku_exp(s, kind, level, thr):
    names = ["det_pku", "det_main"]
    need = [n for n in names if not core.available(n)]
    if need:
        return missing(need)
    test = core.apply_blur(s["test"], kind, level, s["id"])
    o1 = core.keep(cphoto("det_main", s, test, kind, level), thr)
    o2 = core.keep(cphoto("det_pku", s, test, kind, level), thr)
    figs = [fig(draw(test, o1, s["boxes"], width=3), "DeepPCB model, no PKU training",
                short_counts(core.match(o1, s["boxes"], s["labels"]))),
            fig(draw(test, o2, s["boxes"], width=3), "After fine-tuning on PKU",
                short_counts(core.match(o2, s["boxes"], s["labels"])))]
    overall = table([["MGRE-Net (base paper)", "96.4"], ["Ours, DeepPCB model only (zero-shot)", "5.5"],
                     ["<b>Ours, fine-tuned on PKU</b>", "<b>99.24</b>"]], ["model", "mAP@0.5 on PKU test photos (%)"])
    return (card(1, "Real colour photo, before and after PKU training",
                 "Black-and-white DeepPCB scans look nothing like colour photos (a domain gap), so the model needs a "
                 "short fine-tuning on PKU. Red = model output, green = true defects.", grid(*figs))
            + card(2, "On the PKU test photos (random 8:1:1 split)",
                   "Per class AP: open 100, short 100, mousebite 98.6, spur 97.4, spurious copper 99.5, pin-hole 100.", overall))


def fit(img, longest):
    img = np.ascontiguousarray(img[..., :3])
    f = longest / max(img.shape[:2])
    if abs(f - 1) < 0.02:
        return img
    return cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)),
                      interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


def upload_pair(up_test, up_temp, photo):
    t = fit(up_test, 1500 if photo else 640)
    r = cv2.resize(np.ascontiguousarray(up_temp[..., :3]), (t.shape[1], t.shape[0]), interpolation=cv2.INTER_AREA)
    return dict(id="upload", test=t, temp=r, boxes=[], labels=[])


def on_run(mode, exp, source, sel, up_test, up_temp, kind, level, thr, photo=False):
    if source == SRC_UP:
        if up_test is None or up_temp is None:
            return note("Please upload both images: the board to inspect <b>and</b> its defect-free template. "
                        "The system finds defects by comparing the two, so the template is required.")
        s = upload_pair(up_test, up_temp, photo)
        known = False
    else:
        if sel is None:
            return note("Please pick an image first.")
        s = sample(source, sel)
        known = True
    try:
        if mode == MODE_SYS:
            if source == SRC_PKU or (source == SRC_UP and photo):
                return run_pku_system(s, kind, level, thr, known)
            return run_system(s, kind, level, thr, known)
        if exp == EXP_PKU:
            if source != SRC_PKU:
                return note("This experiment uses PKU real photos. Pick one in step 2.")
            return run_pku_exp(s, kind, level, thr)
        if source == SRC_PKU:
            return note("This experiment uses DeepPCB images. Pick one in step 2.")
        if not known and exp != EXP_TPL:
            return note("Experiments compare against the true defects, so they need a dataset image, not an upload.")
        if exp == EXP_ABL:
            return run_ablation(s, kind, level, thr)
        if exp == EXP_BLUR:
            if kind == "None":
                return note("Choose a blur type in step 3 to compare the four methods.")
            return run_blur(s, kind, level, thr)
        return run_template(s, kind, level, thr)
    except Exception as e:
        return note(f"Something went wrong: {html.escape(str(e))}")


def desc_for(mode, exp):
    if mode == MODE_SYS:
        return ("**Our final system:** deblur + binarize the image, compare it with the template, then find and "
                "name every defect.")
    return {EXP_ABL: "Shows how much each part adds: plain detector, + NNVR, + template comparison.",
            EXP_BLUR: "Shows four ways to handle a blurry image, side by side.",
            EXP_TPL: "Cleans the same blurry image with and without the template.",
            EXP_PKU: "Tests on real colour photos from a second dataset (PKU-Market-PCB)."}[exp]


CSS = """
.gradio-container {max-width: 1360px !important; margin: auto;}
#hdr h1 {font-size: 26px; margin: 4px 0 2px 0; font-weight: 700; color: #111827;}
#hdr p {margin: 0; color: #4b5563; font-size: 15px;}
.stephead {font-weight: 600; font-size: 15px; color: #1f2937; margin: 6px 0 -4px 0;}
#runbtn {font-size: 16px; height: 46px;}
.card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px 18px; margin: 0 0 14px 0; background: #fff;}
.card .head {display: flex; gap: 12px; align-items: flex-start; margin-bottom: 10px;}
.card .num {flex: none; width: 28px; height: 28px; border-radius: 50%; background: #2563eb; color: #fff;
  font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: 14px;}
.card h3 {margin: 0; font-size: 17px; color: #111827;}
.card p {margin: 2px 0 0 0; color: #4b5563; font-size: 14px; line-height: 1.45;}
.grid {display: grid; gap: 12px;}
.g1 {grid-template-columns: minmax(0, 520px);} .g2 {grid-template-columns: repeat(2, minmax(0, 1fr));}
.g3 {grid-template-columns: repeat(3, minmax(0, 1fr));} .g4 {grid-template-columns: repeat(4, minmax(0, 1fr));}
@media (max-width: 900px) {.g3, .g4 {grid-template-columns: repeat(2, minmax(0, 1fr));}}
figure {margin: 0;} figure img {width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; display: block;}
figcaption {font-size: 13px; color: #111827; margin-top: 6px;} figcaption .sub {color: #6b7280; margin-top: 2px;}
.chips {display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 12px 0;}
.chip {border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px 14px; min-width: 104px; background: #f9fafb;}
.chip span {display: block; font-size: 20px; font-weight: 700; color: #111827;}
.chip small {color: #6b7280; font-size: 12px;}
.chip.good span {color: #15803d;} .chip.bad span {color: #b91c1c;}
.tblwrap {max-height: 300px; overflow-y: auto; margin-top: 10px;}
.tbl {border-collapse: collapse; width: 100%; font-size: 13.5px; margin-top: 10px;}
.tbl th {text-align: left; background: #f3f4f6; color: #374151; font-weight: 600; padding: 7px 10px;}
.tbl td {border-top: 1px solid #e5e7eb; padding: 7px 10px; color: #111827;}
.ok {color: #15803d; font-weight: 600;} .no {color: #b91c1c; font-weight: 600;}
.note {background: #fffbeb; border: 1px solid #fde68a; color: #78350f; border-radius: 10px; padding: 12px 14px;
  margin-bottom: 14px; font-size: 14px;}
.empty {border: 1.5px dashed #d1d5db; border-radius: 12px; padding: 70px 20px; text-align: center; background: #fafafa;}
.empty .big {font-size: 17px; color: #374151; margin-bottom: 6px;}
.muted {color: #6b7280; font-size: 14px;}
footer {display: none !important;}
"""
