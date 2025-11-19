# latvecs.py
"""
Lattice-vector extraction from a single grain .gro file (from Step 3).

Logic follows Sonali's original notebook/script:
  1) Read .gro, build Site objects
  2) Compute all pairwise COM distances + orientation types (ff/ef/unknown)
  3) Feature engineering and DBSCAN clustering
  4) For each cluster: mean distance, mean axis (with sign fix), majority type
  5) Among the 5 shortest-distance clusters, pick the shortest ff and shortest ef
  6) Save these as lattice vectors to an output .txt file

You typically call `analyze_grain_latvecs` on each *_g1.gro / *_g2.gro from Step 3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from .site import Site
from .iofile import read_gro
from .grotools import split_gro2residues

try:
    import py3Dmol as p3d
except ImportError:
    p3d = None


# =========================
# STEP 4: Feature engineering for clustering
# =========================
def v_quadratic_features(V):
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
    distances,
    vectors,
    cos_orient,
    lam,
    use_invariant=True,
    alpha=1.0,
    beta=0.5,
    zscore=True,
):
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


def cluster_DBSCAN(X, eps=0.6, min_samples=10):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return db.labels_


def _build_sites_from_gro(gro_path):
    """Read .gro and build Site objects + COM array."""
    gro = read_gro(str(gro_path))
    atom_line = gro[2:-1]
    residues = split_gro2residues(atom_line)
    sites = [Site(residue) for residue in residues]
    coms = np.array([site.com for site in sites])
    return sites, coms


def _visualize_coms_py3Dmol(sites, width=400, height=400):
    """Optional py3Dmol visualization of COMs."""
    if p3d is None:
        print("[latvecs] py3Dmol not installed; skipping visualization.")
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


def _pairwise_df(sites):
    """Original STEP 3: compute pairwise distances + PCA alignment and type."""
    df = pd.DataFrame(columns=["res_id1", "res_id2", "distance", "vector", "type"])
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

            new_row = pd.DataFrame(
                [
                    {
                        "res_id1": site1.resid,
                        "res_id2": site2.resid,
                        "distance": dist,
                        "vector": vector,
                        "type": types,
                    }
                ]
            )
            df = pd.concat([df, new_row], ignore_index=True)

    print("len dist_vectors:", len(df))
    return df


def _cluster_and_extract_vectors(df, lam=4.0, eps=0.6, min_samples=5, top_k=5):
    """
    Original STEP 5 + STEP 6 logic:
      - build features
      - DBSCAN
      - per-cluster mean distance, axis, type
      - from the top_k shortest clusters, pick shortest ff and shortest ef
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
    print("Cluster labels:", labels)

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

        # Type assignment
        unique, counts = np.unique(Cos_orient, return_counts=True)
        majority_type = unique[np.argmax(counts)]
        cluster_info[k]["Type"] = "ff" if majority_type == 1 else "ef"

    # STEP 6: Select clusters and visualize
    if not cluster_info:
        print("[latvecs] No non-noise clusters found.")
        return [], cluster_info

    cluster_dists = np.array([info["Distance"] for info in cluster_info.values()])
    # mapping index in this array -> cluster key order
    cluster_keys = list(cluster_info.keys())
    # indices of top_k shortest distances
    k_eff = min(top_k, len(cluster_dists))
    shortest_dists_idx = np.argpartition(cluster_dists, k_eff - 1)[:k_eff]
    shortest_clusters = [cluster_keys[i] for i in shortest_dists_idx]

    # 1) Print all ff/ef among the top_k shortest
    ff_ef_indices = []
    for cid in shortest_clusters:
        ctype = cluster_info[cid]["Type"]
        if ctype in ["ff", "ef"]:
            ff_ef_indices.append(cid)
            print(f"Cluster {cid}: Distance = {cluster_info[cid]['Distance']}, Type = {ctype}")

    # 2) From these, keep only the ff and ef with the *shortest* distance
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

    return list(zip(axis_idx, axis_type, axis_dist, axis)), cluster_info


def analyze_grain_latvecs(
    gro_path,
    output_txt=None,
    visualize=False,
    lam=4.0,
    eps=0.6,
    min_samples=5,
    top_k=5,
):
    """
    Run the full lattice-vector analysis for a single grain .gro.

    Parameters
    ----------
    gro_path : str or Path
        Path to the grain .gro file (e.g. *_g1.gro or *_g2.gro from Step 3).
    output_txt : str or Path, optional
        Output text file. If None, uses "<stem>_output.txt" next to the .gro.
    visualize : bool, default False
        If True and py3Dmol is installed, opens a small COM visualization.
    lam, eps, min_samples, top_k : float/int
        Parameters passed into the original feature-building / DBSCAN logic.

    Returns
    -------
    selected : list of (idx, type, distance, vector)
        The chosen ff/ef clusters (at most 2).
    cluster_info : dict
        All cluster information {cluster_id: {"vector", "Distance", "Type"}}.
    """
    gro_path = Path(gro_path)

    print(f"[latvecs] Reading grain from {gro_path}")
    sites, coms = _build_sites_from_gro(gro_path)
    print(f"Total number of sites: {len(sites)}")
    print(f"Center of Mass calculated for {len(sites)} sites.")

    if visualize:
        view = _visualize_coms_py3Dmol(sites)
        if view is not None:
            view.show()

    # STEP 3: pairwise analysis (in original script numbering)
    df = _pairwise_df(sites)

    # STEP 5 & 6: clustering and vector extraction
    selected, cluster_info = _cluster_and_extract_vectors(
        df,
        lam=lam,
        eps=eps,
        min_samples=min_samples,
        top_k=top_k,
    )

    # Default output file: <stem>_output.txt in same directory
    if output_txt is None:
        output_txt = gro_path.with_name(f"{gro_path.stem}_output.txt")
    else:
        output_txt = Path(output_txt)

    # Save shortest ff/ef vectors (with cluster index and distance)
    with open(output_txt, "w") as f:
        f.write("# idx type distance vec_x vec_y vec_z\n")
        for cid, t, d, vec in selected:
            v = np.asarray(vec).ravel()
            vec_str = " ".join(f"{x:.6f}" for x in v)
            f.write(f"{cid} {t} {d:.6f} {vec_str}\n")

    print(f"Saved shortest ff/ef vectors (with cluster index and distance) to {output_txt}")
    return selected, cluster_info
