#new version which segreggates the grains and gb based on residue mwpca components and gyration long axis
# step2.py — refactored for clarity, same algorithmic logic and outputs
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import MDAnalysis as mda
from sklearn.cluster import KMeans

from src.utils import (
    pbc_radius_neighbors,
    gyration_long_axis,
    nematic_embed,
    nematic_center_to_director,
)

logger = logging.getLogger(__name__)

Label = Literal["grain1", "grain2", "GB"]


def _load_universe(slab_gro: str | Path) -> mda.Universe:
    """Load a slab .gro file into an MDAnalysis Universe."""
    return mda.Universe(str(slab_gro))


def _compute_features(u: mda.Universe, orient_mode: str = "gyration_long"):
    """
    Compute basic per-residue properties used by the segmentation algorithm.

    Parameters
    ----------
    u : MDAnalysis.Universe
        Slab universe.
    orient_mode : str
        How to define the per-residue orientation vector `dirs` used for
        nematic embedding and KMeans. Allowed values (case-insensitive):

            - 'gyration_long'      : original behavior (gyration_long_axis)
            - 'long'               : mwPCA long axis (largest variance)
            - 'short', 'small'     : mwPCA short axis
            - 'normal'             : mwPCA smallest-variance axis (plane normal)
            - 'short_plus_normal',
              'small_plus_normal'  : sum of short + normal axes

        For any mwPCA-based mode, if mwPCA fails (too few atoms, numerical
        issues), the code falls back to 'gyration_long' for that residue.
    """
    mode = orient_mode.lower()
    allowed = {
        "gyration_long",
        "long",
        "short",
        "small",
        "normal",
        "short_plus_normal",
        "small_plus_normal",
    }
    if mode not in allowed:
        raise ValueError(
            f"Unknown orient_mode='{orient_mode}'. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    residues = list(u.residues)
    N = len(residues)
    resids = np.array([r.resid for r in residues], dtype=int)
    COM = np.array([r.atoms.center_of_mass() for r in residues])

    # --- Compute orientation vectors dirs[i] ---
    dirs = np.zeros((N, 3), dtype=float)

    if mode == "gyration_long":
        # Original behavior
        for i, r in enumerate(residues):
            dirs[i] = gyration_long_axis(r.atoms)
    else:
        # mwPCA-based orientations with fallback to gyration_long_axis
        heavy_all = u.select_atoms("not name H*")

        for i, r in enumerate(residues):
            # start with a safe fallback
            fallback_dir = gyration_long_axis(r.atoms)

            # heavy-atom subset of this residue
            ag = r.atoms & heavy_all
            if len(ag) < 6:
                ag = r.atoms.select_atoms("not name H*")
            if len(ag) < 3:
                dirs[i] = fallback_dir
                continue

            # COM for PCA
            com = ag.center_of_mass()
            coords = ag.positions - com  # (n_atoms, 3)
            if coords.shape[0] < 3:
                dirs[i] = fallback_dir
                continue

            masses = ag.masses
            # mass-weighted coordinates
            if masses is None or np.allclose(masses, masses[0]):
                X = coords
            else:
                m_sqrt = np.sqrt(masses).reshape(-1, 1)
                X = coords * m_sqrt

            cov = X.T @ X  # (3,3)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
            except np.linalg.LinAlgError:
                dirs[i] = fallback_dir
                continue

            # Sort eigenvectors by descending eigenvalue
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]
            long_axis = eigvecs[:, 0]
            short_axis = eigvecs[:, 1]
            normal_axis = eigvecs[:, 2]

            if mode == "long":
                v = long_axis
            elif mode in ("short", "small"):
                v = short_axis
            elif mode == "normal":
                v = normal_axis
            elif mode in ("short_plus_normal", "small_plus_normal"):
                v = short_axis + normal_axis
            else:
                # should not happen; keep fallback
                dirs[i] = fallback_dir
                continue

            norm = np.linalg.norm(v)
            if norm < 1e-8:
                # degenerate → fallback
                dirs[i] = fallback_dir
            else:
                dirs[i] = v / norm

    # 6D headless embedding for nematic KMeans
    feat6 = np.array([nematic_embed(d) for d in dirs])

    return residues, N, resids, COM, dirs, feat6


def _smooth_labels_once(lbls: np.ndarray, neigh, maxcos, diff, th_high, margin):
    """Single iteration of smoothing (neighbor voting) — identical logic to original."""
    new = lbls.copy()
    for i, nb in enumerate(neigh):
        if nb.size == 0:
            continue

        counts = {"grain1": 0, "grain2": 0, "GB": 0}
        for j in nb:
            counts[lbls[j]] += 1

        dom = max(counts, key=counts.get)
        tot = sum(counts.values())

        # adopt dominant grain only when conditions match (same as original)
        if dom != "GB" and counts[dom] >= 0.6 * max(tot, 1):
            if not ((maxcos[i] < th_high) or (diff[i] < margin)):
                new[i] = dom
    return new


def _find_boundary_sites(labels, neigh_edge, opp_min_nb):
    """Identify boundary (potential GB) sites using opposite-grain neighbors."""
    N = labels.shape[0]
    is_boundary = np.zeros(N, dtype=bool)
    g1_idx = set(np.where(labels == "grain1")[0])
    g2_idx = set(np.where(labels == "grain2")[0])

    for i in range(N):
        nb = neigh_edge[i]
        if labels[i] == "grain1":
            if sum((j in g2_idx) for j in nb) >= opp_min_nb:
                is_boundary[i] = True
        elif labels[i] == "grain2":
            if sum((j in g1_idx) for j in nb) >= opp_min_nb:
                is_boundary[i] = True
        else:
            # label is already GB
            is_boundary[i] = True
    return is_boundary


def _dilate_mask(mask, neigh, steps: int):
    """Dilate boolean mask by neighbor connectivity for given steps."""
    GB_mask = mask.copy()
    for _ in range(max(0, steps)):
        expand = GB_mask.copy()
        for i in np.where(GB_mask)[0]:
            for j in neigh[i]:
                expand[j] = True
        GB_mask = expand
    return GB_mask


def _remove_small_gb_components(GB_mask, neigh, labels, min_gb_size):
    """Reassign tiny GB connected components to neighboring grains (same logic as original)."""
    if min_gb_size <= 0:
        return GB_mask, labels

    N = labels.shape[0]
    visited = np.zeros(N, dtype=bool)
    for i in range(N):
        if visited[i] or not GB_mask[i]:
            continue

        comp = []
        q = deque([i])
        visited[i] = True

        while q:
            k = q.popleft()
            comp.append(k)
            for j in neigh[k]:
                if GB_mask[j] and not visited[j]:
                    visited[j] = True
                    q.append(j)

        if len(comp) < min_gb_size:
            votes = {"grain1": 0, "grain2": 0}
            for k in comp:
                for j in neigh[k]:
                    if labels[j] in votes:
                        votes[labels[j]] += 1

            dom = "grain1" if votes["grain1"] >= votes["grain2"] else "grain2"
            for k in comp:
                GB_mask[k] = False
                labels[k] = dom

    return GB_mask, labels


def _write_resid_lists(slab_dir: Path, out_prefix: str, slab_gro: Path, resids, final_labels):
    out_g1 = slab_dir / f"{out_prefix}_{slab_gro.stem}_grain1.txt"
    out_gb = slab_dir / f"{out_prefix}_{slab_gro.stem}_GB.txt"
    out_g2 = slab_dir / f"{out_prefix}_{slab_gro.stem}_grain2.txt"

    np.savetxt(out_g1, resids[final_labels == "grain1"], fmt="%d")
    np.savetxt(out_gb, resids[final_labels == "GB"], fmt="%d")
    np.savetxt(out_g2, resids[final_labels == "grain2"], fmt="%d")

    return out_g1, out_gb, out_g2


def step2_clustering_for_slab(
    slab_gro: str | Path,
    slab_dir: str | Path,
    out_prefix: str,
    th_high: float,
    margin: float,
    smooth_iters: int,
    connect_radius: float,
    edge_radius: float,
    opp_min_nb: int,
    edge_dilate_steps: int,
    min_gb_size: int,
    orient_mode: str = "gyration_long",
) -> Dict[str, object]:
    """
    Step 2 — classify slab residues into grain1 / GB / grain2 (topological).

    Behavior, output files and return structure are unchanged from the original
    code, but you can now choose how to define residue orientations via
    `orient_mode`. See `_compute_features` for allowed values.
    """
    slab_gro = Path(slab_gro)
    slab_dir = Path(slab_dir)

    # Load universe and compute per-residue features
    u = _load_universe(slab_gro)
    residues, N, resids, COM, dirs, feat6 = _compute_features(u, orient_mode=orient_mode)
    box = u.dimensions

    # 1) KMeans in nematic-embedded space → two grain directions g1, g2
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(feat6)
    g1 = nematic_center_to_director(km.cluster_centers_[0])
    g2 = nematic_center_to_director(km.cluster_centers_[1])

    # deterministic ordering
    if tuple(np.round(g2, 6)) < tuple(np.round(g1, 6)):
        g1, g2 = g2, g1

    # absolute cosines for headless alignment
    cos1 = np.abs(dirs @ g1)
    cos2 = np.abs(dirs @ g2)
    maxcos = np.maximum(cos1, cos2)
    diff = np.abs(cos1 - cos2)

    # initial labeling
    labels: np.ndarray = np.array(["GB"] * N, dtype=object)
    labels[(cos1 >= th_high) & (cos1 - cos2 >= margin)] = "grain1"
    labels[(cos2 >= th_high) & (cos2 - cos1 >= margin)] = "grain2"

    # neighbor lists under PBC
    neigh = pbc_radius_neighbors(COM, connect_radius, box)
    neigh_edge = pbc_radius_neighbors(COM, edge_radius, box)

    # smoothing
    for _ in range(max(0, smooth_iters)):
        labels = _smooth_labels_once(labels, neigh, maxcos, diff, th_high, margin)

    # identify boundary sites
    is_boundary = _find_boundary_sites(labels, neigh_edge, opp_min_nb)

    # dilation -> GB_mask
    GB_mask = _dilate_mask(is_boundary, neigh, edge_dilate_steps)

    # remove tiny GB components
    GB_mask, labels = _remove_small_gb_components(GB_mask, neigh, labels, min_gb_size)

    # final labels
    final = labels.copy()
    final[GB_mask] = "GB"

    # write outputs
    out_g1, out_gb, out_g2 = _write_resid_lists(slab_dir, out_prefix, slab_gro, resids, final)

    n_g1 = int(np.sum(final == "grain1"))
    n_gb = int(np.sum(final == "GB"))
    n_g2 = int(np.sum(final == "grain2"))

    logger.info(
        "[step2] %s: grain1=%d  GB=%d  grain2=%d  (orient_mode=%s)",
        slab_gro.name,
        n_g1,
        n_gb,
        n_g2,
        orient_mode,
    )

    # Return fresh Universe (like original) and paths
    return {
        "paths": {"g1": out_g1, "gb": out_gb, "g2": out_g2},
        "u": mda.Universe(str(slab_gro)),
    }












#old version which segreggates the grains and gb based on gyration long axis
# # step2.py — refactored for clarity, same algorithmic logic and outputs
# from __future__ import annotations

# import logging
# from collections import deque
# from pathlib import Path
# from typing import Dict, Literal

# import numpy as np
# import MDAnalysis as mda
# from sklearn.cluster import KMeans

# from src.utils import (
#     pbc_radius_neighbors,
#     gyration_long_axis,
#     nematic_embed,
#     nematic_center_to_director,
# )

# logger = logging.getLogger(__name__)

# Label = Literal["grain1", "grain2", "GB"]


# def _load_universe(slab_gro: str | Path) -> mda.Universe:
#     """Load a slab .gro file into an MDAnalysis Universe."""
#     return mda.Universe(str(slab_gro))


# def _compute_features(u: mda.Universe):
#     """Compute basic per-residue properties used by the segmentation algorithm."""
#     residues = list(u.residues)
#     N = len(residues)
#     resids = np.array([r.resid for r in residues], dtype=int)
#     COM = np.array([r.atoms.center_of_mass() for r in residues])
#     dirs = np.array([gyration_long_axis(r.atoms) for r in residues])  # unit vectors
#     feat6 = np.array([nematic_embed(d) for d in dirs])  # 6D headless embedding
#     return residues, N, resids, COM, dirs, feat6


# def _smooth_labels_once(lbls: np.ndarray, neigh, maxcos, diff, th_high, margin):
#     """Single iteration of smoothing (neighbor voting) — identical logic to original."""
#     new = lbls.copy()
#     for i, nb in enumerate(neigh):
#         if nb.size == 0:
#             continue

#         counts = {"grain1": 0, "grain2": 0, "GB": 0}
#         for j in nb:
#             counts[lbls[j]] += 1

#         dom = max(counts, key=counts.get)
#         tot = sum(counts.values())

#         # adopt dominant grain only when conditions match (same as original)
#         if dom != "GB" and counts[dom] >= 0.6 * max(tot, 1):
#             if not ((maxcos[i] < th_high) or (diff[i] < margin)):
#                 new[i] = dom
#     return new


# def _find_boundary_sites(labels, neigh_edge, opp_min_nb):
#     """Identify boundary (potential GB) sites using opposite-grain neighbors."""
#     N = labels.shape[0]
#     is_boundary = np.zeros(N, dtype=bool)
#     g1_idx = set(np.where(labels == "grain1")[0])
#     g2_idx = set(np.where(labels == "grain2")[0])

#     for i in range(N):
#         nb = neigh_edge[i]
#         if labels[i] == "grain1":
#             if sum((j in g2_idx) for j in nb) >= opp_min_nb:
#                 is_boundary[i] = True
#         elif labels[i] == "grain2":
#             if sum((j in g1_idx) for j in nb) >= opp_min_nb:
#                 is_boundary[i] = True
#         else:
#             # label is already GB
#             is_boundary[i] = True
#     return is_boundary


# def _dilate_mask(mask, neigh, steps: int):
#     """Dilate boolean mask by neighbor connectivity for given steps."""
#     GB_mask = mask.copy()
#     for _ in range(max(0, steps)):
#         expand = GB_mask.copy()
#         for i in np.where(GB_mask)[0]:
#             for j in neigh[i]:
#                 expand[j] = True
#         GB_mask = expand
#     return GB_mask


# def _remove_small_gb_components(GB_mask, neigh, labels, min_gb_size):
#     """Reassign tiny GB connected components to neighboring grains (same logic as original)."""
#     if min_gb_size <= 0:
#         return GB_mask, labels

#     N = labels.shape[0]
#     visited = np.zeros(N, dtype=bool)
#     for i in range(N):
#         if visited[i] or not GB_mask[i]:
#             continue

#         comp = []
#         q = deque([i])
#         visited[i] = True

#         while q:
#             k = q.popleft()
#             comp.append(k)
#             for j in neigh[k]:
#                 if GB_mask[j] and not visited[j]:
#                     visited[j] = True
#                     q.append(j)

#         if len(comp) < min_gb_size:
#             votes = {"grain1": 0, "grain2": 0}
#             for k in comp:
#                 for j in neigh[k]:
#                     if labels[j] in votes:
#                         votes[labels[j]] += 1

#             dom = "grain1" if votes["grain1"] >= votes["grain2"] else "grain2"
#             for k in comp:
#                 GB_mask[k] = False
#                 labels[k] = dom

#     return GB_mask, labels


# def _write_resid_lists(slab_dir: Path, out_prefix: str, slab_gro: Path, resids, final_labels):
#     out_g1 = slab_dir / f"{out_prefix}_{slab_gro.stem}_grain1.txt"
#     out_gb = slab_dir / f"{out_prefix}_{slab_gro.stem}_GB.txt"
#     out_g2 = slab_dir / f"{out_prefix}_{slab_gro.stem}_grain2.txt"

#     np.savetxt(out_g1, resids[final_labels == "grain1"], fmt="%d")
#     np.savetxt(out_gb, resids[final_labels == "GB"], fmt="%d")
#     np.savetxt(out_g2, resids[final_labels == "grain2"], fmt="%d")

#     return out_g1, out_gb, out_g2


# def step2_clustering_for_slab(
#     slab_gro: str | Path,
#     slab_dir: str | Path,
#     out_prefix: str,
#     th_high: float,
#     margin: float,
#     smooth_iters: int,
#     connect_radius: float,
#     edge_radius: float,
#     opp_min_nb: int,
#     edge_dilate_steps: int,
#     min_gb_size: int,
# ) -> Dict[str, object]:
#     """
#     Step 2 — classify slab residues into grain1 / GB / grain2 (topological).
#     Behavior, output files and return structure are unchanged from the original code.
#     """
#     slab_gro = Path(slab_gro)
#     slab_dir = Path(slab_dir)

#     # Load universe and compute per-residue features
#     u = _load_universe(slab_gro)
#     residues, N, resids, COM, dirs, feat6 = _compute_features(u)
#     box = u.dimensions

#     # 1) KMeans in nematic-embedded space → two grain directions g1, g2
#     km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(feat6)
#     g1 = nematic_center_to_director(km.cluster_centers_[0])
#     g2 = nematic_center_to_director(km.cluster_centers_[1])

#     # deterministic ordering
#     if tuple(np.round(g2, 6)) < tuple(np.round(g1, 6)):
#         g1, g2 = g2, g1

#     # absolute cosines for headless alignment
#     cos1 = np.abs(dirs @ g1)
#     cos2 = np.abs(dirs @ g2)
#     maxcos = np.maximum(cos1, cos2)
#     diff = np.abs(cos1 - cos2)

#     # initial labeling
#     labels: np.ndarray = np.array(["GB"] * N, dtype=object)
#     labels[(cos1 >= th_high) & (cos1 - cos2 >= margin)] = "grain1"
#     labels[(cos2 >= th_high) & (cos2 - cos1 >= margin)] = "grain2"

#     # neighbor lists under PBC
#     neigh = pbc_radius_neighbors(COM, connect_radius, box)
#     neigh_edge = pbc_radius_neighbors(COM, edge_radius, box)

#     # smoothing
#     for _ in range(max(0, smooth_iters)):
#         labels = _smooth_labels_once(labels, neigh, maxcos, diff, th_high, margin)

#     # identify boundary sites
#     is_boundary = _find_boundary_sites(labels, neigh_edge, opp_min_nb)

#     # dilation -> GB_mask
#     GB_mask = _dilate_mask(is_boundary, neigh, edge_dilate_steps)

#     # remove tiny GB components
#     GB_mask, labels = _remove_small_gb_components(GB_mask, neigh, labels, min_gb_size)

#     # final labels
#     final = labels.copy()
#     final[GB_mask] = "GB"

#     # write outputs
#     out_g1, out_gb, out_g2 = _write_resid_lists(slab_dir, out_prefix, slab_gro, resids, final)

#     n_g1 = int(np.sum(final == "grain1"))
#     n_gb = int(np.sum(final == "GB"))
#     n_g2 = int(np.sum(final == "grain2"))

#     logger.info("[step2] %s: grain1=%d  GB=%d  grain2=%d", slab_gro.name, n_g1, n_gb, n_g2)

#     # Return fresh Universe (like original) and paths (Paths are returned, consistent with original)
#     return {"paths": {"g1": out_g1, "gb": out_gb, "g2": out_g2}, "u": mda.Universe(str(slab_gro))}