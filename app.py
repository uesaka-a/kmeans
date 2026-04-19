import streamlit as st
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cat Clustering", layout="wide")

st.title("Cat Image Clustering")

# クラスタ数
k = st.slider("クラスタ数 k", 1, 5, 3)

# ---------------------------
# データ読み込み（改善版：表示は高画質）
# ---------------------------
@st.cache_data
def load_images(folder):
    data = []
    images = []      # ← 高画質表示用
    filenames = []

    files = os.listdir(folder)[:300]  # ← 多すぎ防止

    for file in files:
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue

        images.append(img)  # ← 元画像を保存（ここが重要）

        # 学習用は小さく
        img_resized = cv2.resize(img, (64, 64))
        img_flat = img_resized.flatten()

        data.append(img_flat)
        filenames.append(file)

    return np.array(data), images, filenames

X, images, filenames = load_images("data/cat")

# ---------------------------
# k-means
# ---------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# ---------------------------
# 代表画像
# ---------------------------
st.markdown("## 各クラスタの代表画像")

centers = kmeans.cluster_centers_
rep_cols = st.columns(k)

for cluster_id in range(k):
    indices = [i for i in range(len(X)) if labels[i] == cluster_id]

    if len(indices) == 0:
        rep_cols[cluster_id].write("なし")
        continue

    distances = []
    for i in indices:
        dist = np.linalg.norm(X[i] - centers[cluster_id])
        distances.append((dist, i))

    _, best_idx = min(distances)

    img = cv2.cvtColor(images[best_idx], cv2.COLOR_BGR2RGB)
    rep_cols[cluster_id].image(img, caption=f"Cluster {cluster_id}", use_column_width=True)

# ---------------------------
# PCA可視化
# ---------------------------
st.markdown("---")
st.markdown("## PCAによる2次元可視化")

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

fig = plt.figure()

for cluster_id in range(k):
    xs = X_2d[labels == cluster_id, 0]
    ys = X_2d[labels == cluster_id, 1]
    plt.scatter(xs, ys, label=f"{cluster_id}")

plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")

st.pyplot(fig)

# ---------------------------
# クラスタごとの画像
# ---------------------------
st.markdown("---")
st.markdown("## クラスタごとの画像")

for cluster_id in range(k):
    st.subheader(f"Cluster {cluster_id}")

    cols = st.columns(5)
    idx = 0

    for i in range(len(images)):
        if labels[i] == cluster_id:
            img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            cols[idx % 5].image(img, width=150)
            idx += 1

    if idx == 0:
        st.write("（画像なし）")