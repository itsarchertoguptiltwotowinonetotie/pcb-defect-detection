import random
import numpy as np
import cv2
import streamlit as st

import pcb_core as core
import pipelines as P

st.set_page_config(page_title="PCB Defect Inspector", page_icon="🔍", layout="wide")

EXTRA_CSS = """
.block-container {max-width: 1320px; padding-top: 4.2rem; padding-bottom: 3rem;}
#hdr {padding: 0 0 14px 0; margin-bottom: 4px; border-bottom: 1px solid #e5e7eb;}
#hdr h1 {font-size: 26px; line-height: 1.25; margin: 0; padding: 0; font-weight: 700; color: #111827;}
#hdr p {margin: 4px 0 0 0; color: #4b5563; font-size: 15px;}
.sec {font-size: 12px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: #2563eb; margin: 2px 0 -8px 0;}
.reshead {display: flex; justify-content: space-between; align-items: baseline; margin: 0 0 10px 0;}
.reshead h2 {font-size: 18px; margin: 0; padding: 0; color: #111827; font-weight: 700;}
.reshead span {color: #6b7280; font-size: 13.5px;}
.empty {padding: 56px 28px; text-align: left;}
.empty ol {margin: 10px auto 0 auto; padding-left: 20px; color: #374151; font-size: 15px; line-height: 1.9; display: inline-block;}
.empty {text-align: center;}
[data-testid="stImageCaption"] {font-size: 12.5px;}
.card figure img {max-height: 360px; object-fit: contain; background: #fff;}
.card .g2 figure img {max-height: 320px;}
"""
st.html(f"<style>{P.CSS}{EXTRA_CSS}</style>")
st.html('<div id="hdr"><h1>🔍 PCB Defect Inspector</h1><p>Finds open, short, mousebite, spur, spurious copper and pin-hole '
        'defects by comparing a board image with its defect-free template, even when the image is blurry.</p></div>')

PLACEHOLDER = ('<div class="empty"><div class="big">No results yet</div>'
               '<ol><li>Choose what you want to see</li><li>Pick an image or upload your own</li>'
               '<li>Optionally add blur, then press <b>Run</b></li></ol></div>')


@st.cache_resource(show_spinner="Getting the models ready (only on the first start, about a minute)...")
def setup():
    core.ensure_weights()
    if core.available("final"):
        core.get("final")
    return True


setup()

SCAN, PHOTO = "Black-and-white scan", "Colour photo"
SHORT = {P.SRC_DEEP: "DeepPCB", P.SRC_PKU: "PKU photos", P.SRC_UP: "Upload"}
FILE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


def choice(label, options, key, fmt=str):
    v = st.segmented_control(label, options, default=options[0], key=key, format_func=fmt, width="stretch")
    return v if v in options else options[0]


def read_upload(f):
    if f is None:
        return None
    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def label_for(data, i):
    it = data[i]
    return f"{it['id']}  ·  " + ", ".join(sorted({P.NAMES[l] for l in it["labels"]}))


left, right = st.columns([5, 8], gap="large")

with left:
    with st.container(border=True):
        st.html('<div class="sec">1 · What to show</div>')
        mode = choice("Mode", [P.MODE_SYS, P.MODE_EXP], "mode")
        exp = st.selectbox("Experiment", P.EXPS, key="exp") if mode == P.MODE_EXP else None
        st.caption(P.desc_for(mode, exp))

        st.html('<div class="sec">2 · Image</div>')
        if mode == P.MODE_SYS:
            sources = [P.SRC_DEEP, P.SRC_PKU, P.SRC_UP]
        elif exp == P.EXP_PKU:
            sources = [P.SRC_PKU]
        elif exp == P.EXP_TPL:
            sources = [P.SRC_DEEP, P.SRC_UP]
        else:
            sources = [P.SRC_DEEP]
        source = choice("Source", sources, f"src_{mode}_{exp}", fmt=lambda s: SHORT[s])

        sel, up_test, up_temp, photo, img_label = None, None, None, False, ""
        if source != P.SRC_UP:
            data = P.PKU if source == P.SRC_PKU else P.DEEP
            c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
            defect = c1.selectbox("Defect type", ["All types"] + P.NAMES[1:], key=f"def_{source}")
            idxs = [i for i, it in enumerate(data) if defect == "All types" or defect in {P.NAMES[l] for l in it["labels"]}]
            pick_key = f"pick_{source}_{defect}"
            if c2.button("Random", width="stretch", icon=":material/shuffle:") and idxs:
                st.session_state[pick_key] = random.choice(idxs)
            if idxs:
                sel = st.selectbox(f"Image ({len(idxs)} available)", idxs, key=pick_key,
                                   format_func=lambda i: label_for(data, i))
                s = P.sample(source, sel)
                img_label = f"{SHORT[source]} {s['id']}"
                p1, p2 = st.columns(2)
                p1.image(s["test"], caption="Test image", width="stretch")
                p2.image(s["temp"], caption="Template (defect-free)", width="stretch")
            else:
                st.info("No images with this defect type.")
        else:
            if mode == P.MODE_SYS:
                photo = choice("Image type", [SCAN, PHOTO], "imgtype") == PHOTO
            u1, u2 = st.columns(2)
            f_test = u1.file_uploader("Board to inspect", type=FILE_TYPES)
            f_temp = u2.file_uploader("Defect-free template", type=FILE_TYPES)
            up_test, up_temp = read_upload(f_test), read_upload(f_temp)
            if up_test is not None:
                u1.image(up_test, width="stretch")
            if up_temp is not None:
                u2.image(up_temp, width="stretch")
            st.caption("Both images are required, from the same board and lined up with each other.")
            img_label = "your upload"

        no_blur_default = source in (P.SRC_PKU, P.SRC_UP) or exp in (P.EXP_ABL, P.EXP_PKU)
        st.html('<div class="sec">3 · Camera blur (optional)</div>')
        ctx = f"{mode}_{exp}_{source}"
        b1, b2 = st.columns([3, 2])
        kind = b1.selectbox("Blur type", core.BLUR_TYPES, index=0 if no_blur_default else 1, key=f"kind_{ctx}",
                            help="Adds blur to the test image to show how the system copes. "
                                 "Keep 'None' if your image is already blurry.")
        level = b2.slider("Level", 1, 5, 4, key=f"lvl_{ctx}", disabled=kind == "None" or "unseen" in kind)
        with st.expander("Advanced settings"):
            thr = st.slider("Confidence threshold", 0.05, 0.95, 0.5 if mode == P.MODE_EXP else 0.8, 0.05,
                            key=f"thr_{mode}", help="Boxes below this confidence are hidden. Our system uses 0.8.")
        run = st.button("Run", type="primary", width="stretch", icon=":material/play_arrow:")

with right:
    if run:
        with st.spinner("Running the models..."):
            html = P.on_run(mode, exp, source, sel, up_test, up_temp, kind, level, thr, photo)
        what = exp if mode == P.MODE_EXP else P.MODE_SYS
        st.session_state["result"] = (ctx, html, f"{what} · {img_label} · {P.blur_text(kind, level)}")
    res = st.session_state.get("result")
    if res and res[0] == ctx:
        st.html(f'<div class="reshead"><h2>Results</h2><span>{res[2]}</span></div>' + res[1])
    else:
        st.html('<div class="reshead"><h2>Results</h2></div>' + PLACEHOLDER)
