import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Klasifikasi Biota Laut", page_icon="🐠", layout="wide")
st.title("🐠 Klasifikasi Biota Laut")
st.caption("CNN & transfer learning MobileNetV2 | Dataset biota laut: pelabelan CLIP + review manual")

BASE = Path(__file__).resolve().parent
HASIL = BASE / "hasil"
MODEL_FILES = {
    "MobileNetV2 (transfer learning)": HASIL / "model_mobilenetv2.keras",
    "CNN dari nol": HASIL / "model_cnn_scratch.keras",
}

# ---------------------------------------------------------------- muat TensorFlow
with st.spinner("Memuat TensorFlow (pertama kali bisa 30-60 detik)..."):
    try:
        import tensorflow as tf
    except Exception as e:
        st.error("TensorFlow gagal dimuat di Python yang menjalankan Streamlit ini.")
        st.exception(e)
        st.stop()

import matplotlib


# ---------------------------------------------------------------- fungsi bantu
@st.cache_data
def muat_info():
    f = HASIL / "info_model.json"
    if f.exists():
        d = json.load(open(f, encoding="utf-8"))
        return d["kelas"], int(d.get("img_size", 160))
    folder = HASIL / "data_berlabel"
    if folder.exists():
        return sorted(p.name for p in folder.iterdir() if p.is_dir()), 160
    return None, None


@st.cache_resource
def muat_model(path_str):
    return tf.keras.models.load_model(path_str, compile=False)


@st.cache_resource
def siapkan_gradcam(path_str):
    m = muat_model(path_str)
    base = None
    for l in m.layers:
        if isinstance(l, tf.keras.Model) and any(x.name == "out_relu" for x in l.layers):
            base = l
            break
    if base is None:
        return None
    feat = tf.keras.Model(base.input, base.get_layer("out_relu").output)
    return feat, m.layers[-1]


def siapkan_gambar(img, size):
    if img.mode in ("P", "LA", "RGBA"):
        img = img.convert("RGBA").convert("RGB")
    else:
        img = img.convert("RGB")
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype="float32") / 255.0


def hitung_gradcam(feat, head, arr, kelas, size):
    x = tf.convert_to_tensor(arr[None] * 2.0 - 1.0, dtype=tf.float32)
    with tf.GradientTape() as tape:
        fmap = feat(x, training=False)
        tape.watch(fmap)
        preds = head(tf.reduce_mean(fmap, axis=(1, 2)))
        skor = preds[:, kelas]
    g = tape.gradient(skor, fmap)
    w = tf.reduce_mean(g, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(fmap[0] * w, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., None], (size, size))[..., 0].numpy()


def tumpuk_heatmap(arr, cam, alpha=0.45):
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    out = np.clip((1 - alpha) * arr + alpha * heat, 0, 1)
    return (out * 255).astype("uint8")