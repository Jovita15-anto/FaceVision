import io

import numpy as np
import streamlit as st
from PIL import Image

from facevision.detector import annotate, classify, crop, detect_raw, load_detector

st.set_page_config(page_title="FaceVision", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');
:root { --amber:#FFB547; --edge:#2A3B57; }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
.block-container { max-width:1100px; padding-top:2.5rem; }
.hero h1 { font-size:3rem; font-weight:800; letter-spacing:-0.03em; margin:0; }
.hero p { color:#9FB0C8; font-size:1.1rem; max-width:34rem; margin:.4rem 0 1.5rem; }
/* Viewfinder corners around every image */
[data-testid="stImage"] {
  padding:12px; margin-bottom:.5rem;
  background:
    linear-gradient(var(--amber),var(--amber)) top left/24px 2px,
    linear-gradient(var(--amber),var(--amber)) top left/2px 24px,
    linear-gradient(var(--amber),var(--amber)) top right/24px 2px,
    linear-gradient(var(--amber),var(--amber)) top right/2px 24px,
    linear-gradient(var(--amber),var(--amber)) bottom left/24px 2px,
    linear-gradient(var(--amber),var(--amber)) bottom left/2px 24px,
    linear-gradient(var(--amber),var(--amber)) bottom right/24px 2px,
    linear-gradient(var(--amber),var(--amber)) bottom right/2px 24px;
  background-repeat:no-repeat;
}
[data-testid="stImage"] img { border-radius:2px; }
[data-testid="stMetric"] { background:#152238; border:1px solid var(--edge); border-radius:10px; padding:1rem 1.2rem; }
[data-testid="stFileUploaderDropzone"] { border:1.5px dashed var(--edge); border-radius:12px; }
button[data-baseweb="tab"] { font-weight:700; }
</style>
<div class="hero">
  <h1>FaceVision</h1>
  <p>Upload a photo to find and count every face, with confidence scores and facial landmarks.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading face detector…")
def get_detector():
    return load_detector()


@st.cache_data(show_spinner="Detecting faces…")
def raw_detect(img):
    return detect_raw(get_detector(), img)


def read_image(file):
    try:
        image = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
        image.thumbnail((1280, 1280))  # large photos make MTCNN very slow
        return np.array(image)
    except Exception as e:
        st.error(f"Couldn't read this image ({e}). Try a JPG or PNG.")
        return None


with st.sidebar:
    st.header("Settings")
    min_conf = st.slider("Minimum confidence", 0.50, 0.999, 0.995, 0.001, format="%.3f",
                         help="Faces scoring below this are ignored.")
    padding = st.slider("Crop padding (px)", 0, 60, 20)
    landmarks = st.toggle("Show landmarks", value=True)
    strict = st.toggle("Filter false positives", value=True,
                       help="Drops detections whose landmarks don't look like a face, and overlapping boxes.")
    debug = st.toggle("Debug: show all detections", value=False)
    st.caption("Powered by MTCNN")

file = st.file_uploader("Drop a photo here", type=["jpg", "jpeg", "png"], key="detect")
img = read_image(file) if file else None
if img is not None:
    results = classify(raw_detect(img), min_conf, strict)
    faces = [f for f, status in results if status == "kept"]
    if debug:
        with st.expander("Debug: all raw detections", expanded=True):
            st.write(f"MTCNN returned {len(results)} candidate(s)")
            st.table([{"#": i + 1, "confidence": f"{f.confidence:.3f}",
                       "box (x, y, w, h)": str(f.box), "status": status}
                      for i, (f, status) in enumerate(results)])
    if not faces:
        st.warning("No faces found. Try lowering the minimum confidence in the sidebar.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Faces found", len(faces))
        c2.metric("Average confidence", f"{np.mean([f.confidence for f in faces]):.1%}")
        st.image(annotate(img, faces, padding, landmarks), use_container_width=True)
        st.subheader("Cropped faces")
        cols = st.columns(4)
        for i, f in enumerate(faces):
            cols[i % 4].image(crop(img, f, padding),
                              caption=f"Face {i + 1} · {f.confidence:.1%}",
                              use_container_width=True)
