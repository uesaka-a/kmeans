import streamlit as st
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Cat Clustering", layout="wide")

# ---------------------------
# CSS
# ---------------------------
st.markdown("""
<style>
/* ラベル */
[data-testid="stWidgetLabel"] p {
    font-size: 24px !important;
    font-weight: bold !important;
}

/* スライダーの数字（目盛り） */
[data-baseweb="slider"] span {
    font-size: 18px !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Cat Image Clustering（色ベース）")

# ---------------------------
# UI
# ---------------------------
k = st.slider("クラスタ数 k", 1, 6, 3)

# ---------------------------
# データ読み込み（中央クロップ＋色ヒストグラム）
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

        # -------- 中央クロップ --------
        h, w, _ = img.shape
        crop = img[h//4:3*h//4, w//4:3*w//4]

        # -------- リサイズ（安定化）--------
        crop = cv2.resize(crop, (64, 64))

        # -------- 色ヒストグラム --------
        hist = cv2.calcHist([crop], [0,1,2], None, [8,8,8], [0,256]*3)
        hist = cv2.normalize(hist, hist).flatten()

        data.append(hist)
        filenames.append(file)

    return np.array(data), images, filenames

# ---------------------------
# 前処理（正規化 → PCA）
# ---------------------------
@st.cache_data
def preprocess(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=50, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    return X_reduced

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
X_reduced = preprocess(X)
kmeans, labels = run_kmeans(X_reduced, k)

# ---------------------------
# クラスタ表示
# ---------------------------
st.markdown("<h3 style='font-size:22px;'>クラスタ（色ベース）</h3>", unsafe_allow_html=True)

centers = kmeans.cluster_centers_

for cluster_id in range(k):
    st.subheader(f"Cluster {cluster_id}")

    indices = np.where(labels == cluster_id)[0]
    st.write(f"枚数: {len(indices)}")

    if len(indices) == 0:
        st.write("（画像なし）")
        continue

    # -------- 距離計算（NumPy）--------
    cluster_points = X_reduced[indices]
    dists = np.linalg.norm(cluster_points - centers[cluster_id], axis=1)
    sorted_idx = np.argsort(dists)

    best_idx = indices[sorted_idx[0]]
    nearest = [indices[i] for i in sorted_idx[1:11]]

    # -------- 代表画像 --------
    img = cv2.cvtColor(images[best_idx], cv2.COLOR_BGR2RGB)
    st.image(img, width=160)

    # -------- 近い画像 --------
    cols = st.columns(10)
    for j, i in enumerate(nearest):
        img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        cols[j].image(img, width=80)

    st.markdown("---")