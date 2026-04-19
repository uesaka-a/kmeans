import streamlit as st
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cat Clustering", layout="wide")

# ---------------------------
# CSS
# ---------------------------
st.markdown("""
<style>
[data-testid="stWidgetLabel"] p {
    font-size: 24px !important;
    font-weight: bold !important;
}
div[data-testid="stSliderTickBar"] span {
    font-size: 16px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("Cat Image Clustering")

# ---------------------------
# UI
# ---------------------------
k = st.slider("クラスタ数 k", 1, 6, 3)

# ---------------------------
# データ読み込み
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

        img_resized = cv2.resize(img, (64, 64))
        img_flat = img_resized.flatten()

        data.append(img_flat)
        filenames.append(file)

    return np.array(data), images, filenames

# ---------------------------
# 前処理（正規化 → PCA）
# ---------------------------
@st.cache_data
def preprocess(X):
    # 正規化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA（50次元）
    pca = PCA(n_components=50, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    return X_scaled, X_reduced, pca

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

# ---------------------------
# PCA可視化用（2次元）
# ---------------------------
@st.cache_data
def run_pca_2d(X):
    pca = PCA(n_components=2)
    return pca.fit_transform(X)

# 実行
X, images, filenames = load_images("data/cat")
X_scaled, X_reduced, pca_model = preprocess(X)

kmeans, labels = run_kmeans(X_reduced, k)
X_2d = run_pca_2d(X_reduced)

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

    # -------- 距離（NumPyで一括）--------
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

# ---------------------------
# PCA可視化
# ---------------------------
st.markdown("<h3 style='font-size:28px;'>PCAによる2次元可視化</h3>", unsafe_allow_html=True)

plt.rcParams.update({'font.size': 3})

fig = plt.figure(figsize=(3, 2))

for cluster_id in range(k):
    xs = X_2d[labels == cluster_id, 0]
    ys = X_2d[labels == cluster_id, 1]
    plt.scatter(xs, ys, label=f"{cluster_id}", s=6)

plt.legend(fontsize=4)
plt.xlabel("PC1", fontsize=4)
plt.ylabel("PC2", fontsize=4)
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)

st.pyplot(fig)