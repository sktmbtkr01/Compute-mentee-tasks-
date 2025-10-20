# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import sys

st.set_page_config(layout="wide", page_title="Clustering Playground", initial_sidebar_state="collapsed")

# ---- Try import HDBSCAN or mark unavailable ----
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

# ---- Styling ----
st.markdown(
    """
    <style>
    .reportview-container .main .block-container{padding-top:1rem;}
    .title {font-size:28px; font-weight:700; margin-bottom:0.1rem;}
    .subtitle {color: #6c757d; margin-top:0;}
    .control-box {background: #ffffff; padding: 12px; border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.06);}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header ----
st.markdown('<div class="title">Clustering Playground — Synthetic Datasets & Visualizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Create datasets, tune preprocessing & clustering hyperparameters, and inspect results interactively (Plotly)</div>', unsafe_allow_html=True)
st.write("---")

# ---- Controls container ----
with st.container():
    left_col, right_col = st.columns([1.6, 2.4])

    with left_col:
        st.markdown("#### Dataset")
        with st.form(key="dataset_form"):
            dataset_type = st.selectbox("Choose synthetic dataset", ["make_blobs", "make_moons"])
            n_samples = st.slider("Number of samples", min_value=200, max_value=2000, value=1000, step=100)
            random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
            if dataset_type == "make_blobs":
                n_features = st.slider("Number of features (dimensions)", min_value=2, max_value=8, value=2)
                centers_option = st.selectbox("Centers option", ["auto", "int: 2-6", "array-like (circular)"])
                if centers_option.startswith("int"):
                    n_centers = st.slider("Number of centers (clusters)", 2, 8, 4)
                else:
                    n_centers = 4
                cluster_std = st.slider("Cluster standard deviation", 0.1, 3.0, 0.6)
            else:  # make_moons
                n_features = 2  # make_moons only creates 2D
                n_centers = None
                cluster_std = st.slider("Noise (for make_moons)", 0.0, 0.5, 0.1)
            submitted_dataset = st.form_submit_button("Generate / Update dataset")

        st.markdown("#### Preprocessing & Feature Scaling")
        scaler_name = st.selectbox("Feature scaler", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])
        apply_pca = st.checkbox("Force 2D/3D visualization using PCA if dims > 3", value=True)
        reduce_to = st.selectbox("Plot dimension", [2, 3], index=0)

    with right_col:
        st.markdown("#### Algorithm & Hyperparameters")
        algo = st.selectbox("Clustering algorithm", 
                            options=[opt for opt in ["KMeans", "DBSCAN", "Agglomerative", "HDBSCAN"] if (HDBSCAN_AVAILABLE or opt != "HDBSCAN")])
        st.write("Tune algorithm-specific hyperparameters below.")
        # Dynamic widgets for algorithm
        if algo == "KMeans":
            k = st.slider("n_clusters (k)", 2, 10, 4)
            km_init = st.selectbox("init", ["k-means++", "random"])
            km_n_init = st.slider("n_init", 1, 20, 10)
            km_max_iter = st.number_input("max_iter", min_value=100, max_value=2000, value=300)
            show_centroids = st.checkbox("Show centroids (2D/3D only)", value=True)
        elif algo == "DBSCAN":
            eps = st.slider("eps", 0.01, 5.0, 0.5, step=0.01)
            db_min_samples = st.slider("min_samples", 1, 50, 5)
            metric = st.selectbox("metric", ["euclidean", "manhattan", "cosine", "l2"])
        elif algo == "Agglomerative":
            agg_k = st.slider("n_clusters", 2, 10, 4)
            linkage = st.selectbox("linkage", ["ward", "complete", "average", "single"])
            affinity = None
            if linkage != "ward":
                affinity = st.selectbox("affinity (distance metric)", ["euclidean", "l1", "l2", "manhattan", "cosine"])
        elif algo == "HDBSCAN":
            if not HDBSCAN_AVAILABLE:
                st.error("hdbscan package is not installed in this environment. Install with `pip install hdbscan` to enable.")
                st.stop()
            h_min_cluster_size = st.slider("min_cluster_size", 2, 200, 5)
            h_min_samples = st.slider("min_samples (use 0 for automatic)", 0, 200, 5)
            h_metric = st.selectbox("metric", ["euclidean", "manhattan", "cosine", "l2"])

        st.markdown("#### Plot options")
        show_outliers_as_marker = st.checkbox("Highlight noise/outliers (DBSCAN/HDBSCAN) as 'x'", value=True)
        marker_size = st.slider("Marker size", 4, 16, 8)

# ---- Dataset generation function ----
@st.cache_data
def generate_dataset(dtype, n_samples, n_features, centers, cluster_std, random_state):
    if dtype == "make_blobs":
        # if centers passed as int, use that, else auto
        if isinstance(centers, int):
            X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                              cluster_std=cluster_std, random_state=random_state)
        else:
            # default centers
            X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=4,
                              cluster_std=cluster_std, random_state=random_state)
    elif dtype == "make_moons":
        X, y = make_moons(n_samples=n_samples, noise=cluster_std, random_state=random_state)
    else:
        X = np.zeros((n_samples, 2))
        y = np.zeros(n_samples, dtype=int)
    return X, y

# ---- Build dataset ----
# ---- Build dataset (robust generation) ----
# Ensure variable n_centers exists (it may only be created inside the form branch)
try:
    n_centers
except NameError:
    n_centers = None

# Always generate a dataset (use current widget values) so X is always defined
if dataset_type == "make_blobs":
    centers_param = n_centers if isinstance(n_centers, int) else None
    X, y_true = generate_dataset("make_blobs", n_samples, n_features, centers_param, cluster_std, random_state)
else:
    # make_moons only produces 2D; generator ignores n_features, centers
    X, y_true = generate_dataset("make_moons", n_samples, 2, None, cluster_std, random_state)

# ---- Scaling ----
if scaler_name != "None":
    if scaler_name == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_name == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler is not None:
        try:
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            X_scaled = X.copy()
    else:
        X_scaled = X.copy()
else:
    # no scaler selected
    X_scaled = X.copy()


# ---- Scaling ----
if scaler_name != "None":
    if scaler_name == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_name == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler is not None:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.copy()
else:
    X_scaled = X.copy()

# ---- If features > 3 and user asked to reduce for plotting, apply PCA for visualization only ----
plot_X = X_scaled.copy()
original_dim = plot_X.shape[1]
if reduce_to in (2, 3):
    if plot_X.shape[1] > reduce_to:
        if apply_pca:
            pca = PCA(n_components=reduce_to, random_state=random_state)
            plot_X = pca.fit_transform(plot_X)
        else:
            # pick first dims if user doesn't want PCA
            plot_X = plot_X[:, :reduce_to]
    elif plot_X.shape[1] < reduce_to:
        # pad with zeros to satisfy 3D plotting if needed
        pad_width = reduce_to - plot_X.shape[1]
        plot_X = np.hstack([plot_X, np.zeros((plot_X.shape[0], pad_width))])

# ---- Fit chosen algorithm ----
cluster_labels = None
clusterer_info = ""
try:
    if algo == "KMeans":
        km = KMeans(n_clusters=k, init=km_init, n_init=km_n_init, max_iter=km_max_iter, random_state=random_state)
        cluster_labels = km.fit_predict(X_scaled)
        clusterer_info = f"KMeans (k={k})"
        centroids = km.cluster_centers_
    elif algo == "DBSCAN":
        db = DBSCAN(eps=eps, min_samples=db_min_samples, metric=metric)
        cluster_labels = db.fit_predict(X_scaled)
        clusterer_info = f"DBSCAN (eps={eps:.2f}, min_samples={db_min_samples})"
        centroids = None
    elif algo == "Agglomerative":
        # some sklearn versions ignore affinity parameter for 'ward'
        if linkage == "ward":
            agg = AgglomerativeClustering(n_clusters=agg_k, linkage=linkage)
        else:
            agg = AgglomerativeClustering(n_clusters=agg_k, linkage=linkage, affinity=affinity)
        cluster_labels = agg.fit_predict(X_scaled)
        clusterer_info = f"Agglomerative (k={agg_k}, linkage={linkage})"
        centroids = None
    elif algo == "HDBSCAN":
        # HDBSCAN returns -1 for noise like DBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=h_min_cluster_size, min_samples=(None if h_min_samples == 0 else h_min_samples),
                                   metric=h_metric)
        cluster_labels = clusterer.fit_predict(X_scaled)
        clusterer_info = f"HDBSCAN (min_cluster_size={h_min_cluster_size}, min_samples={h_min_samples})"
        centroids = None
    else:
        cluster_labels = np.zeros(X_scaled.shape[0], dtype=int)
        centroids = None
except Exception as e:
    st.error(f"Error fitting algorithm: {e}")
    st.stop()

# ---- Metrics ----
# count clusters ignoring noise label -1
unique_labels = set(cluster_labels.tolist())
n_clusters_found = len([lab for lab in unique_labels if lab != -1])
# silhouette only valid when at least 2 clusters (and not all noise)
sil_score = None
try:
    if n_clusters_found >= 2:
        # Silhouette requires labels not all the same and not a single cluster
        valid_mask = cluster_labels != -1
        if valid_mask.sum() >= 2 and len(set(cluster_labels[valid_mask])) >= 2:
            sil_score = silhouette_score(X_scaled[valid_mask], cluster_labels[valid_mask])
except Exception:
    sil_score = None

# ---- Prepare DataFrame for plotting ----
if reduce_to == 2:
    df_plot = pd.DataFrame(plot_X, columns=["x", "y"])
elif reduce_to == 3:
    df_plot = pd.DataFrame(plot_X, columns=["x", "y", "z"])
else:
    df_plot = pd.DataFrame(plot_X, columns=[f"f{i}" for i in range(plot_X.shape[1])])

df_plot["cluster"] = cluster_labels.astype(int).astype(str)  # string for color grouping
df_plot["label_true"] = y_true if y_true is not None else -1
df_plot["idx"] = np.arange(len(df_plot))

# map noise label '-1' to a visible group
# Plotly: create categorical coloring
# create hover template
hover_cols = []
if reduce_to == 2:
    hover_cols = ["x", "y", "cluster", "label_true", "idx"]
elif reduce_to == 3:
    hover_cols = ["x", "y", "z", "cluster", "label_true", "idx"]

# ---- Top summary ----
left_summary, right_summary = st.columns([2, 1])
with left_summary:
    st.markdown("#### Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", f"{X.shape[0]:,}")
    c2.metric("Original dims", f"{original_dim}")
    c3.metric("Clusters found", f"{n_clusters_found}")
    st.write(f"**Algorithm:** {clusterer_info}")
    if sil_score is not None:
        st.success(f"Silhouette score: {sil_score:.3f}")
    else:
        st.info("Silhouette score: Not available (need ≥2 clusters with non-noise points)")

with right_summary:
    st.markdown("#### Notes")
    st.write("- Noise/outliers are labeled `-1` for DBSCAN/HDBSCAN.")
    st.write("- If dataset dims > plot dims, PCA is used for visualization (if enabled).")
    if not HDBSCAN_AVAILABLE:
        st.write("- HDBSCAN not available in this environment; to enable install `hdbscan`.")
    st.write("")

# ---- Plotting ----
st.markdown("---")
st.markdown("#### Interactive Plot")

if reduce_to == 2:
    fig = px.scatter(df_plot, x="x", y="y", color="cluster", labels={"cluster": "cluster"},
                     hover_data=hover_cols, height=650)
    # highlight noise marker style
    if show_outliers_as_marker:
        # alter marker symbol for cluster -1 points
        noise_idx = df_plot['cluster'] == "-1"
        if noise_idx.any():
            # add trace for noise with 'x' marker
            noise_df = df_plot[noise_idx]
            fig.add_scatter(x=noise_df["x"], y=noise_df["y"], mode="markers",
                            marker=dict(symbol="x", size=marker_size + 2, line=dict(width=1)),
                            name="noise/outlier", hoverinfo="skip")
    # centroids
    if algo == "KMeans" and show_centroids:
        if centroids is not None:
            # centroids may be in original scaled space; for plotting we need to transform centroids to PCA space if used
            centroids_plot = centroids.copy()
            if original_dim != 2 and plot_X.shape[1] == 2:
                if apply_pca and original_dim > 2:
                    # transform centroids by same PCA used
                    try:
                        centroids_plot = pca.transform(centroids)
                    except Exception:
                        # fallback: take first two dims
                        centroids_plot = centroids_plot[:, :2]
            fig.add_scatter(x=centroids_plot[:, 0], y=centroids_plot[:, 1], mode="markers",
                            marker=dict(symbol="diamond", size=12, line=dict(width=1.5)),
                            name="centroids")
    fig.update_traces(marker=dict(size=marker_size))
    st.plotly_chart(fig, use_container_width=True)

elif reduce_to == 3:
    fig = px.scatter_3d(df_plot, x="x", y="y", z="z", color="cluster", hover_data=hover_cols, height=720)
    fig.update_traces(marker=dict(size=marker_size))
    # Noisy points and centroids similar handling:
    if show_outliers_as_marker:
        noise_df = df_plot[df_plot['cluster'] == "-1"]
        if not noise_df.empty:
            fig.add_scatter3d(x=noise_df["x"], y=noise_df["y"], z=noise_df["z"], mode="markers",
                              marker=dict(symbol="x", size=marker_size + 2),
                              name="noise/outlier", hoverinfo="skip")
    if algo == "KMeans" and show_centroids and centroids is not None:
        centroids_plot = centroids.copy()
        if centroids_plot.shape[1] != 3:
            if apply_pca and original_dim > 3:
                centroids_plot = pca.transform(centroids)
            else:
                # pad or slice
                if centroids_plot.shape[1] < 3:
                    pad = 3 - centroids_plot.shape[1]
                    centroids_plot = np.hstack([centroids_plot, np.zeros((centroids_plot.shape[0], pad))])
                else:
                    centroids_plot = centroids_plot[:, :3]
        fig.add_scatter3d(x=centroids_plot[:, 0], y=centroids_plot[:, 1], z=centroids_plot[:, 2], mode="markers",
                          marker=dict(symbol="diamond", size=12), name="centroids")
    st.plotly_chart(fig, use_container_width=True)

# ---- Cluster table & counts ----
st.markdown("---")
st.markdown("#### Cluster breakdown & sample preview")
cluster_summary = df_plot.groupby("cluster").size().reset_index(name="count").sort_values("count", ascending=False)
st.dataframe(cluster_summary, use_container_width=True)

st.markdown("Sample points (first 10 rows):")
st.dataframe(df_plot.head(10), use_container_width=True)

# ---- Footer / tips ----
st.markdown("---")
st.markdown("#### Tips")
st.markdown("""
- If you want clearer cluster separation for `make_blobs`, reduce `cluster_std`.
- For DBSCAN, `eps` and `min_samples` tuning is important; try smaller `eps` or higher `min_samples` to reduce noise.
- HDBSCAN finds variable-density clusters; try increasing `min_cluster_size` to merge small clusters.
- Use PCA forcing to visualize high-dimensional blobs in 2D/3D.
""")
