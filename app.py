import streamlit as st
import os
import cv2
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cat Clustering", layout="wide")

st.title("🐱 Cat Image Clustering")

# ---------------------------
# UI
# ---------------------------
k = st.slider("クラスタ数 k", 1, 10, 3)

# ---------------------------
# データ読み込み（高画質＋キャッシュ）
# ---------------------------
@st.cache_data
def load_images(folder):
    data = []
    images = []
    filenames = []

    files = os.listdir(folder)[:300]

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
# k-means（キャッシュ）
# ---------------------------
@st.cache_data
def run_kmeans(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

# ---------------------------
# PCA（キャッシュ）
# ---------------------------
@st.cache_data
def run_pca(X):
    pca = PCA(n_components=2)
    return pca.fit_transform(X)

# 実行
X, images, filenames = load_images("data/cat")
kmeans, labels = run_kmeans(X, k)
X_2d = run_pca(X)

# ---------------------------
# 代表＋ランダム5枚
# ---------------------------
st.markdown("## 🧠 クラスタ代表＋サンプル")

centers = kmeans.cluster_centers_

for cluster_id in range(k):
    st.subheader(f"Cluster {cluster_id}")

    indices = [i for i in range(len(X)) if labels[i] == cluster_id]

    if len(indices) == 0:
        st.write("（画像なし）")
        continue

    # ---- 代表画像 ----
    distances = []
    for i in indices:
        dist = np.linalg.norm(X[i] - centers[cluster_id])
        distances.append((dist, i))

    _, best_idx = min(distances)

    col1, col2 = st.columns([1, 4])

    with col1:
        img = cv2.cvtColor(images[best_idx], cv2.COLOR_BGR2RGB)
        st.image(img, caption="代表", width=120)

    # ---- ランダム5枚（代表除外） ----
    candidate_indices = [i for i in indices if i != best_idx]

    sample = random.sample(candidate_indices, min(5, len(candidate_indices)))

    with col2:
        cols = st.columns(5)
        for j, i in enumerate(sample):
            img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            cols[j].image(img, width=120)

# ---------------------------
# PCA可視化
# ---------------------------
st.markdown("---")
st.markdown("## 📉 PCAによる2次元可視化")

fig = plt.figure()

for cluster_id in range(k):
    xs = X_2d[labels == cluster_id, 0]
    ys = X_2d[labels == cluster_id, 1]
    plt.scatter(xs, ys, label=f"{cluster_id}")

plt.legend()
plt.xlabel("PC1")
plt.ylabel("PC2")

st.pyplot(fig)