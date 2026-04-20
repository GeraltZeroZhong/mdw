from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from deeptime.decomposition import TICA
except Exception:
    TICA = None

from ...config import AdvancedAnalysisConfig
from ...core import save_csv


def save_projection_per_replica(replica_dirs, feature_list, all_proj, prefix, out_dir, max_saved_components):
    start = 0
    for replica_dir, feat in zip(replica_dirs, feature_list):
        n = feat.shape[0]
        proj = all_proj[start:start+n]
        rows = [[i, *proj[i, :min(max_saved_components, proj.shape[1])].tolist()] for i in range(n)]
        save_csv(
            Path(out_dir) / f"{replica_dir.name}_{prefix}.csv",
            ["frame"] + [f"{prefix.upper()}{i+1}" for i in range(min(max_saved_components, proj.shape[1]))],
            rows,
        )
        start += n


def run_pca(X_all, cfg: AdvancedAnalysisConfig):
    scaler = StandardScaler().fit(X_all)
    X_scaled = scaler.transform(X_all)
    n_pca_use = min(cfg.n_pca, X_scaled.shape[0], X_scaled.shape[1])
    if n_pca_use < 2:
        raise ValueError("PCA 可用维度不足。")
    model = PCA(n_components=n_pca_use, random_state=cfg.random_state)
    transformed = model.fit_transform(X_scaled)
    return scaler, X_scaled, model, transformed


def run_tica(scaled_feature_list, cfg: AdvancedAnalysisConfig):
    if TICA is None:
        raise ImportError("缺少 deeptime.decomposition.TICA。")
    if not scaled_feature_list:
        raise ValueError("No feature trajectories were provided for tICA.")
    min_frames = min(arr.shape[0] for arr in scaled_feature_list)
    safe_lag = min(cfg.tica_lag_frames, max(1, min_frames - 1))
    effective_pairs = sum(max(0, arr.shape[0] - safe_lag) for arr in scaled_feature_list)
    n_tics_use = min(cfg.n_tics, scaled_feature_list[0].shape[1], effective_pairs)
    if n_tics_use < 2:
        raise ValueError("tICA 可用维度不足。")
    model = TICA(lagtime=safe_lag, dim=n_tics_use).fit(scaled_feature_list).fetch_model()
    transformed_list = [model.transform(arr) for arr in scaled_feature_list]
    transformed = np.vstack(transformed_list)
    return safe_lag, model, transformed


def run_clustering(X_tica, cfg: AdvancedAnalysisConfig):
    embed = X_tica[:, :min(3, X_tica.shape[1])]
    n_clusters_use = min(cfg.n_clusters, embed.shape[0])
    if n_clusters_use < 2:
        raise ValueError("聚类可用簇数不足。")
    kmeans = KMeans(n_clusters=n_clusters_use, random_state=cfg.random_state, n_init=20)
    labels = kmeans.fit_predict(embed)
    return embed, n_clusters_use, kmeans, labels


def build_dtrajs_from_labels(replica_dirs, feature_list, labels):
    trajectories = []
    start = 0
    for replica_dir, feat in zip(replica_dirs, feature_list):
        n = feat.shape[0]
        trajectories.append(np.asarray(labels[start:start+n], dtype=np.int32).ravel())
        start += n
    return trajectories
