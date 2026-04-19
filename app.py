import streamlit as st
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

st.set_page_config(page_title="Cat Clustering", layout="wide")

# ---------------------------
# CSS（スライダー改善）
# ---------------------------
st.markdown("""
<style>

/* ラベル */
[data-testid="stWidgetLabel"] p {
    font-size: 26px !important;
    font-weight: bold !important;
}

/* 目盛り */
[data-baseweb="slider"] * {
    font-size: 18px !important;
    font-weight: bold !important;
}

</style>
""", unsafe_allow_html=True)

st.title("Cat Image Clustering")

# ---------------------------
# UI
# ---------------------------
k = st.slider("クラスタ数 k", 1, 6, 3)

# ---------------------------
# データ読み込み（改善版）
# ---------------------------
@st.cache_data
def load_images(folder):
    data = []
    images = []
    filenames = []

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:300]

    for file in files:
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue

        images.append(img)

        h, w, _ = img.shape

        # -------- ① 強めクロップ --------
        crop = img[h//3:2*h//3, w//3:2*w//3]

        crop = cv2.resize(crop, (64, 64))

        # -------- ② 明るさ除去 --------
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = 0
        crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # -------- ③ 粗いヒストグラム --------
        hist = cv2.calcHist([crop], [0,1,2], None, [3,3,3], [0,256]*3)

        # 正規化
        hist = hist / (np.sum(hist) + 1e-6)

        data.append(hist.flatten())
        filenames.append(file)

    return np.array(data), images, filenames

# ---------------------------
# k-means
# ---------------------------
@st.cache_data
def run_kmeans(X, k):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )
    labels = kmeans.fit_predict(X)
    return kmeans, labels

# 実行
X, images, filenames = load_images("data/cat")
kmeans, labels = run_kmeans(X, k)

# ---------------------------
# クラスタ表示
# ---------------------------
st.markdown("<h3 style='font-size:22px;'>クラスタ（改善版）</h3>", unsafe_allow_html=True)

centers = kmeans.cluster_centers_

for cluster_id in range(k):
    st.subheader(f"Cluster {cluster_id}")

    indices = np.where(labels == cluster_id)[0]
    st.write(f"枚数: {len(indices)}")

    if len(indices) == 0:
        st.write("（画像なし）")
        continue

    # 距離計算
    cluster_points = X[indices]
    dists = np.linalg.norm(cluster_points - centers[cluster_id], axis=1)
    sorted_idx = np.argsort(dists)

    best_idx = indices[sorted_idx[0]]
    nearest = [indices[i] for i in sorted_idx[1:11]]

    # 代表画像
    img = cv2.cvtColor(images[best_idx], cv2.COLOR_BGR2RGB)
    st.image(img, width=160)

    # 近い画像
    cols = st.columns(10)
    for j, i in enumerate(nearest):
        img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        cols[j].image(img, width=80)

    st.markdown("---")