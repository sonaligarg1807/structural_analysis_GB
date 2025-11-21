# latvecs.py
"""
Lattice-vector extraction from a single grain .gro file (from Step 3).

This is a reorganized (import-safe) version of the original module:
- preserves the exact computational logic and output format
- adds light documentation, type hints and structured logging
- keeps the same filenames and output behavior so it remains compatible
  with the rest of the workflow.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.grotools import split_gro2residues
from src.iofile import read_gro
from src.site import Site

try:
    import py3Dmol as p3d  # optional visualization
except Exception:
    p3d = None  # type: ignore

logger = logging.getLogger(__name__)

__all__ = ["analyze_grain_latvecs", "_pairwise_df", "_cluster_and_extract_vectors"]


# =========================
# STEP 4: Feature engineering for clustering
# =========================
def v_quadratic_features(V: np.ndarray) -> np.ndarray:
    vx, vy, vz = V[:, 0], V[:, 1], V[:, 2]
    return np.column_stack(
        [
            vx * vx,
            vy * vy,
            vz * vz,
            np.sqrt(2) * vx * vy,
            np.sqrt(2) * vx * vz,
            np.sqrt(2) * vy * vz,
        ]
    )


def build_features(
    distances: Sequence[float],
    vectors: Sequence[Sequence[float]],
    cos_orient: Sequence[float],
    lam: float,
    use_invariant: bool = True,
    alpha: float = 1.0,
    beta: float = 0.5,
    zscore: bool = True,
) -> np.ndarray:
    """
    Build features used for DBSCAN clustering of pairwise site vectors.

    Same logic as the original code (no algorithmic changes).
    """
    D = np.asarray(distances).astype(float)
    V = np.asarray(vectors).astype(float)
    C = np.asarray(cos_orient).astype(float)

    w = np.exp(-D / lam)[:, None]
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Vn = V / norms

    V_feat = v_quadratic_features(Vn) if use_invariant else Vn
    X = np.hstack([w, alpha * V_feat, beta * C[:, None]])

    if zscore:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X


def cluster_DBSCAN(X: np.ndarray, eps: float = 0.6, min_samples: int = 10) -> np.ndarray:
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return db.labels_


def _build_sites_from_gro(gro_path: Path) -> Tuple[List[Site], np.ndarray]:
    """Read .gro and build Site objects + COM array."""
    gro = read_gro(str(gro_path))
    atom_line = gro[2:-1]
    residues = split_gro2residues(atom_line)
    sites = [Site(residue) for residue in residues]
    coms = np.array([site.com for site in sites])
    return sites, coms


def _visualize_coms_py3Dmol(sites: List[Site], width: int = 400, height: int = 400):
    """Optional py3Dmol visualization of COMs."""
    if p3d is None:
        logger.debug("[latvecs] py3Dmol not installed; skipping visualization.")
        return None

    view = p3d.view(width=width, height=height)
    for site in sites:
        com = site.com
        sphere = {
            "center": {"x": float(com[0]), "y": float(com[1]), "z": float(com[2])},
            "radius": 0.2,
            "color": "red",
            "opacity": 0.8,
        }
        view.addSphere(sphere)
    view.setBackgroundColor("0xeeeeee")
    view.zoomTo()
    return view


def _pairwise_df(sites: List[Site]) -> pd.DataFrame:
    """
    Compute pairwise distances, vectors and orientation type between all site pairs.

    IDENTICAL classification logic to original implementation.
    """
    rows_list = []  # Accumulate rows in a list
    resids = [site.resid for site in sites]

    for i in range(len(resids)):
        for j in range(i + 1, len(resids)):
            site1 = sites[i]
            site2 = sites[j]
            com1, com2 = site1.com, site2.com
            vector = np.array(com2) - np.array(com1)
            dist = np.linalg.norm(vector)

            _, _, R1 = site1.mw_pca_axes(n_comps=2, rot="rmat")
            _, _, R2 = site2.mw_pca_axes(n_comps=2, rot="rmat")
            sim_fac = np.dot(R1.T, R2)
            diag_elements = np.diag(sim_fac)

            # Classification logic (unchanged)
            if np.round(diag_elements[0], 1) < 0.9:
                types = "unknown"
            elif all(np.round(diag_elements, 1) > 0.9):
                types = "ff"
            elif np.round(diag_elements[0], 1) > 0.9 and np.round(diag_elements[1], 1) < 0.9:
                types = "ef"
            else:
                types = "unknown"

            # ← CHANGED: Append dict to list instead of DataFrame concat
            rows_list.append({
                "res_id1": site1.resid,
                "res_id2": site2.resid,
                "distance": dist,
                "vector": vector,
                "type": types,
            })

    # Single concat at the end
    df = pd.DataFrame(rows_list)

    logger.debug("len dist_vectors: %d", len(df))
    return df


def _cluster_and_extract_vectors(
    df: pd.DataFrame,
    lam: float = 4.0,
    eps: float = 0.6,
    min_samples: int = 5,
    top_k: int = 5,
):
    """
    Cluster pairwise vectors and extract representative lattice-vector directions.

    Returns (selected_list, cluster_info) exactly as original:
      - selected_list: list of (cluster_idx, type, distance, vector)
      - cluster_info: dict mapping cluster_id -> {"vector","Distance","Type"}
    """
    # STEP 5: Cluster analysis with orientation fix
    dist_values = df["distance"].tolist()
    dot_products = [1 if t == "ff" else 0 for t in df["type"].tolist()]
    vectors = df["vector"].tolist()

    X = build_features(
        dist_values,
        vectors,
        dot_products,
        lam=lam,
        use_invariant=True,
        alpha=1.0,
        beta=0.5,
        zscore=True,
    )
    labels = cluster_DBSCAN(X, eps=eps, min_samples=min_samples)
    logger.debug("Cluster labels: %s", labels)

    cluster_info = {}
    unique_labels = np.unique(labels)
    for k in unique_labels:
        if k == -1:
            # noise
            continue
        idx = labels == k
        cluster_info[int(k)] = {"vector": [], "Distance": None, "Type": None}
        Dists = np.array(dist_values)[idx]
        Vecs = np.array(vectors)[idx]
        Cos_orient = np.array(dot_products)[idx]

        # Mean distance
        cluster_info[k]["Distance"] = np.mean(Dists)

        # Orientation fix: enforce sign consistency
        V = Vecs / np.linalg.norm(Vecs, axis=1, keepdims=True)
        ref = V[0]
        for i in range(len(V)):
            if np.dot(V[i], ref) < 0:
                V[i] *= -1

        # Angular variance check
        T = (V[:, :, None] * V[:, None, :]).mean(axis=0)
        w, e = np.linalg.eigh(T)
        mean_axis = e[:, np.argmax(w)]
        cluster_info[k]["vector"] = mean_axis / np.linalg.norm(mean_axis)

        # Type assignment (keep original majority-vote logic)
        unique, counts = np.unique(Cos_orient, return_counts=True)
        majority_type = unique[np.argmax(counts)]
        cluster_info[k]["Type"] = "ff" if majority_type == 1 else "ef"

    # STEP 6: Select clusters and return results (same selection logic)
    if not cluster_info:
        logger.info("[latvecs] No non-noise clusters found.")
        return [], cluster_info

    cluster_dists = np.array([info["Distance"] for info in cluster_info.values()])
    cluster_keys = list(cluster_info.keys())
    k_eff = min(top_k, len(cluster_dists))
    shortest_dists_idx = np.argpartition(cluster_dists, k_eff - 1)[:k_eff]
    shortest_clusters = [cluster_keys[i] for i in shortest_dists_idx]

    ff_ef_indices = []
    for cid in shortest_clusters:
        ctype = cluster_info[cid]["Type"]
        if ctype in ["ff", "ef"]:
            ff_ef_indices.append(cid)
            logger.debug("Cluster %s: Distance = %s, Type = %s", cid, cluster_info[cid]["Distance"], ctype)

    shortest_ff_idx, shortest_ff_dist = None, np.inf
    shortest_ef_idx, shortest_ef_dist = None, np.inf

    for cid in ff_ef_indices:
        info = cluster_info[cid]
        d = info["Distance"]
        if info["Type"] == "ff" and d < shortest_ff_dist:
            shortest_ff_dist = d
            shortest_ff_idx = cid
        elif info["Type"] == "ef" and d < shortest_ef_dist:
            shortest_ef_dist = d
            shortest_ef_idx = cid

    selected_indices = [i for i in (shortest_ff_idx, shortest_ef_idx) if i is not None]

    axis, axis_type = [], []
    axis_dist, axis_idx = [], []

    for cid in selected_indices:
        info = cluster_info[cid]
        axis.append(info["vector"])
        axis_type.append(info["Type"])
        axis_dist.append(info["Distance"])
        axis_idx.append(cid)

    selected = list(zip(axis_idx, axis_type, axis_dist, axis))
    return selected, cluster_info


def analyze_grain_latvecs(
    gro_path: Path | str,
    output_txt: Optional[Path | str] = None,
    visualize: bool = False,
    lam: float = 4.0,
    eps: float = 0.6,
    min_samples: int = 5,
    top_k: int = 5,
):
    """
    Run the full lattice-vector analysis for a single grain .gro.

    Returns
    -------
    selected : list of (idx, type, distance, vector)
        The chosen ff/ef clusters (at most 2).
    cluster_info : dict
        All cluster information {cluster_id: {"vector", "Distance", "Type"}}.
    """
    gro_path = Path(gro_path)

    logger.info("[latvecs] Reading grain from %s", gro_path)
    sites, coms = _build_sites_from_gro(gro_path)
    logger.info("Total number of sites: %d", len(sites))

    if visualize:
        view = _visualize_coms_py3Dmol(sites)
        if view is not None:
            view.show()

    df = _pairwise_df(sites)

    selected, cluster_info = _cluster_and_extract_vectors(
        df, lam=lam, eps=eps, min_samples=min_samples, top_k=top_k
    )

    if output_txt is None:
        output_txt = gro_path.with_name(f"{gro_path.stem}_latvecs.txt")
    else:
        output_txt = Path(output_txt)

    with open(output_txt, "w") as f:
        f.write("# idx type distance vec_x vec_y vec_z\n")
        for cid, t, d, vec in selected:
            v = np.asarray(vec).ravel()
            vec_str = " ".join(f"{x:.6f}" for x in v)
            f.write(f"{cid} {t} {d:.6f} {vec_str}\n")

    logger.info("Saved shortest ff/ef vectors (with cluster index and distance) to %s", output_txt)
    return selected, cluster_info