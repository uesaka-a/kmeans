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
[data-testid="stWidgetLabel"] p {
    font-size: 26px !important;
    font-weight: bold !important;
}
[data-baseweb="slider"] * {
    font-size: 18px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Cat Image Clustering（安定版）")

# ---------------------------
# UI
# ---------------------------
k = st.slider("クラスタ数 k", 2, 6, 3)  # ← 1は意味ないので2以上に

# ---------------------------
# データ読み込み
# ---------------------------
@st.cache_data
def load_images(folder):
    data = []
    images = []

    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:300]

    for file in files:
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue

        images.append(img)

        h, w, _ = img.shape

        # -------- クロップ（適度）--------
        crop = img[h//2-80:h//2+80, w//2-80:w//2+80]
        crop = cv2.resize(crop, (64, 64))

        # -------- 色ヒストグラム（安定版）--------
        hist = cv2.calcHist([crop], [0,1,2], None, [4,4,4], [0,256]*3)

        # 正規化（割合にする）
        hist = hist / (np.sum(hist) + 1e-6)

        data.append(hist.flatten())

    return np.array(data), images

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
X, images = load_images("data/cat")
kmeans, labels = run_kmeans(X, k)

# ---------------------------
# クラスタ表示
# ---------------------------
st.markdown("<h3 style='font-size:22px;'>クラスタ</h3>", unsafe_allow_html=True)

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