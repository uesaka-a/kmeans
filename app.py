import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

st.set_page_config(page_title="Cat Clustering", layout="wide")

st.title("Cat Image Clustering")

# クラスタ数選択
k = st.slider("クラスタ数 k", 1, 5, 3)

# データ読み込み
@st.cache_data
def load_images(folder):
    data = []
    images = []
    filenames = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue

        img_resized = cv2.resize(img, (64, 64))
        img_flat = img_resized.flatten()

        data.append(img_flat)
        images.append(img_resized)
        filenames.append(file)

    return np.array(data), images, filenames

X, images, filenames = load_images("data/cat")

# k-means
kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
labels = kmeans.fit_predict(X)

# ---------------------------
# 代表画像表示
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
# 全画像クラスタ表示
# ---------------------------
st.markdown("---")
st.markdown("## 📊 クラスタごとの画像")

for cluster_id in range(k):
    st.subheader(f"Cluster {cluster_id}")

    cols = st.columns(5)
    idx = 0

    for i in range(len(images)):
        if labels[i] == cluster_id:
            img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            cols[idx % 5].image(img, use_column_width=True)
            idx += 1

    if idx == 0:
        st.write("（画像なし）")