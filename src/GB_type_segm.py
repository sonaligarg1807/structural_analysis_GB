#remove 3layer selection logic and also removed the mentioning of gb-axis, slab-axis and margin-axis logic. now the code selects molecules based on slab_normal, gb_normal
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Single-layer grain/GB selection in a *local* frame.

This version does NOT rely on cartesian x/y/z from Step 1.
Instead it:

  1) Computes a slab normal n_slab from the geometry of the slab.
  2) Computes a GB normal n_gb (inside the slab plane) from the GB residues.
  3) Builds an orthonormal basis (t, n_gb, n_slab) where:
         s = COM · n_slab     (slab-normal coordinate)
         g = COM · n_gb       (GB-normal coordinate in the slab plane)
         m = COM · t          (transverse coordinate along the GB)

  4) Selects a single mid-layer along s (slab_thick).
  5) Within that layer, segments GB resids into connected components,
     defines a GB band around each segment at g ≈ y0, and defines
     grain bands at offset ±gb_offset along g.
  6) Within each grain band, picks the largest connected component,
     optionally capped at target_per_side using BFS from the centroid.

The output API (rows list, fields) is kept compatible with the main pipeline,
except that `gb_axis` is now just a label passed through (the true geometry
is in the local frame described above).
"""

from __future__ import annotations

import logging
import re
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.utils import (
    margin_mask_1d,
    write_gro_for_resids,
    connected_components,
)

logger = logging.getLogger(__name__)

__all__ = ["step3_single_layer_for_slab"]

# -------------------------
# Local helpers
# -------------------------


def largest_component_within(
    idxs: np.ndarray | List[int],
    COM: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Return indices of the largest connected component within `idxs`.
    Uses COM distances with a given neighbor radius.
    """
    idxs = np.asarray(idxs, dtype=int)
    if idxs.size == 0:
        return np.array([], dtype=int)

    comps = connected_components(COM, radius, subset_idx=idxs)
    if not comps:
        return np.array([], dtype=int)

    comps.sort(key=len, reverse=True)
    return comps[0]


def cap_component(
    ids: np.ndarray | List[int],
    target: Optional[int],
    COM: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Cap a connected component to at most `target` elements by BFS from centroid.

    If target is None or component is already smaller than target,
    return `ids` unchanged.
    """
    ids = np.asarray(ids, dtype=int)
    if target is None or ids.size == 0 or ids.size <= target:
        return ids

    sub = ids
    nbr_local = NearestNeighbors(radius=radius).fit(COM[sub])
    G = nbr_local.radius_neighbors_graph(COM[sub], mode="connectivity")

    ctr = COM[sub].mean(axis=0)
    seed = int(np.argmin(np.linalg.norm(COM[sub] - ctr, axis=1)))

    seen = np.zeros(len(sub), dtype=bool)
    seen[seed] = True
    sel = [seed]
    q: deque[int] = deque([seed])

    while q and len(sel) < target:
        i = q.popleft()
        for j in G[i].indices:
            if not seen[j]:
                seen[j] = True
                q.append(j)
                sel.append(j)
                if len(sel) >= target:
                    break

    return sub[np.array(sel, dtype=int)]


def _load_anysep(path: str | Path) -> set[int]:
    """
    Load integers from a text file with arbitrary separators. Returns a set of ints.
    """
    path = Path(path)
    s: set[int] = set()
    with path.open() as f:
        for line in f:
            for tok in re.findall(r"-?\d+", line):
                s.add(int(tok))
    return s


def _valid_resids(path: str | Path, resid_to_idx: Dict[int, int]) -> List[int]:
    """Filter loaded resids to those present in `resid_to_idx`."""
    resids = _load_anysep(path)
    return sorted(r for r in resids if r in resid_to_idx)


def _pca_frame_from_points(points: np.ndarray) -> np.ndarray:
    """
    PCA on a set of 3D points.

    Returns a 3x3 matrix whose columns are eigenvectors corresponding
    to ascending eigenvalues (smallest variance first).
    """
    pts = np.asarray(points, float)
    if pts.shape[0] < 3:
        # Not enough points for a stable PCA; fall back to identity.
        return np.eye(3)

    ctr = pts.mean(axis=0)
    X = pts - ctr
    cov = np.cov(X.T)
    vals, vecs = np.linalg.eigh(cov)  # eigenvalues ascending
    order = np.argsort(vals)          # explicit, though eigh already returns sorted
    vecs = vecs[:, order]             # columns: v0 (min var), v1, v2 (max var)
    return vecs


def _compute_local_frame(COM: np.ndarray, gb_idx_all: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute an orthonormal basis (t, n_gb, n_slab) for the slab.

    - n_slab : slab normal (smallest variance direction)
    - n_gb   : GB-normal direction in the slab plane (intermediate variance)
    - t      : transverse direction along the GB (largest variance)

    We primarily use GB residues to define this frame. If GB is too small,
    we fall back to all COMs.
    """
    if gb_idx_all.size >= 10:
        pts = COM[gb_idx_all]
        frame = _pca_frame_from_points(pts)
    else:
        logger.warning("STEP3: Not enough GB residues (%d) for stable PCA; using all COMs.", gb_idx_all.size)
        frame = _pca_frame_from_points(COM)

    # columns: v0 (min var), v1, v2 (max var)
    v0 = frame[:, 0]  # slab normal candidate
    v1 = frame[:, 1]  # GB-normal candidate
    v2 = frame[:, 2]  # GB tangent candidate

    # Normalize (eigh should already give unit vectors, but be safe)
    n_slab = v0 / np.linalg.norm(v0)
    n_gb = v1 - np.dot(v1, n_slab) * n_slab
    n_gb /= np.linalg.norm(n_gb)

    t = np.cross(n_slab, n_gb)
    t /= np.linalg.norm(t)

    return t, n_gb, n_slab


# -------------------------
# Step 3 — Single-layer grain/GB selection in local frame
# -------------------------


def step3_single_layer_for_slab(
    slab_gro: str | Path,
    slab_dir: str | Path,
    step2_out: Dict[str, object],
    a_len: float,
    b_len: float,
    c_len: float,
    alpha: float,
    beta: float,
    gamma: float,
    slab_thick: float,
    gb_offset: float,
    gb_band_thick: float,
    box_margin: float,
    connect_radius_layer: float,
    write_txt: bool,
    write_gro: bool,
    min_gb_to_write: int,
    target_per_side: Optional[int],
) -> List[dict]:
    """
    Select a single mid-layer and partition it into GB / grain1 / grain2
    bands using a *local* geometric frame.

    The local frame is constructed as follows:

      - Use GB residue COMs to perform PCA.
      - Smallest-variance eigenvector -> slab normal n_slab.
      - Second eigenvector -> GB-normal direction n_gb in the slab plane.
      - Third eigenvector -> transverse direction t along the GB.

    For each residue i with COM r_i, define:

      s_i = r_i · n_slab    (slab-normal coordinate)
      g_i = r_i · n_gb      (GB-normal coordinate)
      m_i = r_i · t         (transverse coordinate)

    Then:

      - Select a mid-layer in s_i with thickness slab_thick.
      - Within that layer, segment GB into connected components.
      - For each GB segment, define:
          * GB band: |g - y0| <= gb_band_thick/2
          * Grain bands:
              g in [y0 - (gb_offset + gb_band_thick), y0 - gb_offset]
              or   [y0 + gb_offset, y0 + (gb_offset + gb_band_thick)]
      - Within grain bands, pick the largest connected component;
        optionally cap to target_per_side molecules.

    Returns a list of dict rows compatible with the main workflow.
    """
    slab_gro = Path(slab_gro)
    slab_dir = Path(slab_dir)

    # universe / coordinates
    u = step2_out["u"]
    residues = list(u.residues)

    resid_to_idx: Dict[int, int] = {r.resid: i for i, r in enumerate(residues)}
    idx_to_resid: Dict[int, int] = {i: r.resid for i, r in enumerate(residues)}

    # COMs in lab frame (Å)
    COM = np.array([r.atoms.center_of_mass() for r in residues])

    # --- load grain / GB resids from Step 2 paths ---
    paths = step2_out["paths"]
    G1_resids = _valid_resids(paths["g1"], resid_to_idx)
    G2_resids = _valid_resids(paths["g2"], resid_to_idx)
    GB_resids = _valid_resids(paths["gb"], resid_to_idx)

    g1_idx_all = np.array([resid_to_idx[r] for r in G1_resids], dtype=int)
    g2_idx_all = np.array([resid_to_idx[r] for r in G2_resids], dtype=int)
    gb_idx_all = np.array([resid_to_idx[r] for r in GB_resids], dtype=int)

    if gb_idx_all.size == 0:
        logger.warning("STEP3: No GB residues found for %s; returning empty rows.", slab_gro.name)
        return []

    # --- construct local frame (t, n_gb, n_slab) from GB geometry ---
    t, n_gb, n_slab = _compute_local_frame(COM, gb_idx_all)

    # project COMs into local coordinates
    s_coord = COM @ n_slab   # slab-normal
    g_coord = COM @ n_gb     # GB-normal (in-plane)
    m_coord = COM @ t        # transverse

    # mid-layer along slab-normal
    s_min, s_max = float(s_coord.min()), float(s_coord.max())
    slab_center = 0.5 * (s_min + s_max)
    layer_mask_this = np.abs(s_coord - slab_center) <= slab_thick / 2.0

    # margin in transverse direction: keep only central band in m_coord
    margin_mask_axis = margin_mask_1d(m_coord, box_margin)

    logger.info(
        "[step3-single-local] %s: s_range=(%.2f, %.2f), center=%.2f, slab_thick=%.2f",
        slab_gro.name,
        s_min,
        s_max,
        slab_center,
        slab_thick,
    )

    # -----------------------
    # Segment GB within mid-layer
    # -----------------------
    gb_idx_layer = gb_idx_all[layer_mask_this[gb_idx_all] & margin_mask_axis[gb_idx_all]]

    if gb_idx_layer.size:
        nbrs = NearestNeighbors(radius=connect_radius_layer).fit(COM[gb_idx_layer])
        G = nbrs.radius_neighbors_graph(COM[gb_idx_layer], mode="connectivity")
        seen = np.zeros(len(gb_idx_layer), dtype=bool)
        comps: List[np.ndarray] = []

        for s0 in range(len(gb_idx_layer)):
            if seen[s0]:
                continue
            q: deque[int] = deque([s0])
            seen[s0] = True
            cur: List[int] = [s0]
            while q:
                i = q.popleft()
                for j in G[i].indices:
                    if not seen[j]:
                        seen[j] = True
                        q.append(j)
                        cur.append(j)
            comps.append(np.array(cur, dtype=int))

        gb_segments = [gb_idx_layer[c] for c in comps]
    else:
        gb_segments = []

    slab_stem = slab_gro.stem
    rows: List[dict] = []
    seg_id = 0

    # for ranking: box "center" along GB normal in local frame
    g_min, g_max = float(g_coord.min()), float(g_coord.max())
    g_mid = 0.5 * (g_min + g_max)

    for seg in sorted(gb_segments, key=len, reverse=True):
        seg_id += 1

        # GB center along g
        y0 = float(np.median(g_coord[seg]))

        # GB band around y0
        gb_band_mask = (
            (np.abs(g_coord - y0) <= gb_band_thick / 2.0)
            & layer_mask_this
            & margin_mask_axis
        )
        gb_band_ids = np.intersect1d(seg, np.where(gb_band_mask)[0])

        # Grain bands at offset ±gb_offset along g
        lo1, hi1 = y0 - (gb_offset + gb_band_thick), y0 - gb_offset
        lo2, hi2 = y0 + gb_offset, y0 + (gb_offset + gb_band_thick)

        def in_bands(arr: np.ndarray) -> np.ndarray:
            return ((arr >= lo1) & (arr <= hi1)) | ((arr >= lo2) & (arr <= hi2))

        # candidates within mid-layer, margin, and bands
        g1_cand = g1_idx_all[layer_mask_this[g1_idx_all] & margin_mask_axis[g1_idx_all]]
        g1_cand = g1_cand[in_bands(g_coord[g1_cand])]
        if gb_idx_all.size:
            g1_cand = np.setdiff1d(g1_cand, gb_idx_all, assume_unique=False)

        g2_cand = g2_idx_all[layer_mask_this[g2_idx_all] & margin_mask_axis[g2_idx_all]]
        g2_cand = g2_cand[in_bands(g_coord[g2_cand])]
        if gb_idx_all.size:
            g2_cand = np.setdiff1d(g2_cand, gb_idx_all, assume_unique=False)

        # enforce connectedness & optional capping
        g1_cand = largest_component_within(g1_cand, COM, connect_radius_layer)
        g2_cand = largest_component_within(g2_cand, COM, connect_radius_layer)

        if target_per_side:
            g1_cand = cap_component(g1_cand, target_per_side, COM, connect_radius_layer)
            g2_cand = cap_component(g2_cand, target_per_side, COM, connect_radius_layer)

        gb_res = sorted(idx_to_resid[i] for i in gb_band_ids)
        g1_res = sorted(idx_to_resid[i] for i in g1_cand)
        g2_res = sorted(idx_to_resid[i] for i in g2_cand)

        gb_n, g1_n, g2_n = len(gb_res), len(g1_res), len(g2_res)
        dist_center = abs(y0 - g_mid)

        write_paths: Optional[Dict[str, str]] = None
        if (gb_n >= min_gb_to_write) and write_gro:
            stem = f"{slab_stem}_seg{seg_id:02d}_local{y0:.2f}"
            gb_gro = slab_dir / f"{stem}_gb.gro"
            g1_gro = slab_dir / f"{stem}_g1.gro"
            g2_gro = slab_dir / f"{stem}_g2.gro"

            resid_to_idx_cache = {r.resid: i for i, r in enumerate(residues)}
            ok1 = write_gro_for_resids(u, gb_res, str(gb_gro), resid_to_idx_cache)
            ok2 = write_gro_for_resids(u, g1_res, str(g1_gro), resid_to_idx_cache)
            ok3 = write_gro_for_resids(u, g2_res, str(g2_gro), resid_to_idx_cache)

            if ok1 and ok2 and ok3:
                write_paths = {"gb": str(gb_gro), "g1": str(g1_gro), "g2": str(g2_gro)}
                if write_txt:
                    np.savetxt(slab_dir / f"{stem}_gb.txt", np.array(gb_res, int), fmt="%d")
                    np.savetxt(slab_dir / f"{stem}_g1.txt", np.array(g1_res, int), fmt="%d")
                    np.savetxt(slab_dir / f"{stem}_g2.txt", np.array(g2_res, int), fmt="%d")

        rows.append(
            {
                "slab_dir": slab_dir.name,
                "slab_stem": slab_stem,
                "group_id": seg_id,
                "gb_axis": "local",  # label only; true geometry in local frame
                "y_center_A": y0,              # GB center along local n_gb
                "dist_to_box_center_A": dist_center,  # distance from local mid g_mid
                "GB_N": gb_n,
                "G1_N": g1_n,
                "G2_N": g2_n,
                "contact_plane": "NA",
                "paths": write_paths,
            }
        )

    return rows













# # Refactored GB_type_segm.py — preserved algorithm, clearer structure and docstrings
# from __future__ import annotations

# import json
# import logging
# import re
# from collections import deque
# from pathlib import Path
# from typing import Dict, List, Optional

# import numpy as np
# from sklearn.neighbors import NearestNeighbors

# from src.utils import (
#     axis_vals_dict,
#     margin_mask_1d,
#     midlayer_center,
#     layer_mask,
#     unitcell_delta_for_axis,
#     write_gro_for_resids,
#     connected_components,
# )

# logger = logging.getLogger(__name__)

# __all__ = ["step3_single_layer_for_slab", "step3_merge_layers_for_slab"]

# # -------------------------
# # Local helpers
# # -------------------------


# def largest_component_within(idxs: np.ndarray | List[int], COM: np.ndarray, radius: float) -> np.ndarray:
#     """
#     Return indices of the largest connected component within `idxs`.
#     """
#     idxs = np.asarray(idxs, dtype=int)
#     if idxs.size == 0:
#         return np.array([], dtype=int)

#     comps = connected_components(COM, radius, subset_idx=idxs)
#     if not comps:
#         return np.array([], dtype=int)

#     comps.sort(key=len, reverse=True)
#     return comps[0]


# def cap_component(ids: np.ndarray | List[int], target: Optional[int], COM: np.ndarray, radius: float) -> np.ndarray:
#     """
#     Cap a connected component to at most `target` elements by BFS from centroid.
#     If target is None or component smaller than target, return `ids` unchanged.
#     """
#     ids = np.asarray(ids, dtype=int)
#     if target is None or ids.size == 0 or ids.size <= target:
#         return ids

#     sub = ids
#     nbr_local = NearestNeighbors(radius=radius).fit(COM[sub])
#     G = nbr_local.radius_neighbors_graph(COM[sub], mode="connectivity")

#     ctr = COM[sub].mean(axis=0)
#     seed = int(np.argmin(np.linalg.norm(COM[sub] - ctr, axis=1)))

#     seen = np.zeros(len(sub), dtype=bool)
#     seen[seed] = True
#     sel = [seed]
#     q: deque[int] = deque([seed])

#     while q and len(sel) < target:
#         i = q.popleft()
#         for j in G[i].indices:
#             if not seen[j]:
#                 seen[j] = True
#                 q.append(j)
#                 sel.append(j)
#                 if len(sel) >= target:
#                     break

#     return sub[np.array(sel, dtype=int)]


# def _load_anysep(path: str | Path) -> set[int]:
#     """
#     Load integers from a text file with arbitrary separators. Returns a set of ints.
#     """
#     path = Path(path)
#     s: set[int] = set()
#     with path.open() as f:
#         for line in f:
#             for tok in re.findall(r"-?\d+", line):
#                 s.add(int(tok))
#     return s


# def _valid_resids(path: str | Path, resid_to_idx: Dict[int, int]) -> List[int]:
#     """Filter loaded resids to those present in `resid_to_idx`."""
#     resids = _load_anysep(path)
#     return sorted(r for r in resids if r in resid_to_idx)


# # -------------------------
# # Step 3A — Single-layer grain/GB selection
# # -------------------------


# def step3_single_layer_for_slab(
#     slab_gro: str | Path,
#     slab_dir: str | Path,
#     step2_out: Dict[str, object],
#     gb_axis: str,
#     a_len: float,
#     b_len: float,
#     c_len: float,
#     alpha: float,
#     beta: float,
#     gamma: float,
#     slab_thick: float,
#     gb_offset: float,
#     gb_band_thick: float,
#     box_margin: float,
#     connect_radius_layer: float,
#     write_txt: bool,
#     write_gro: bool,
#     min_count_write: int,
#     min_gb_to_write: int,
#     target_per_side: Optional[int],
#     slab_axis: str,
# ) -> List[dict]:
#     """
#     Select the single mid-layer and partition it into GB / grain1 / grain2 bands.

#     Returns a list of dict rows compatible with the main workflow.
#     """
#     slab_gro = Path(slab_gro)
#     slab_dir = Path(slab_dir)

#     # Validate slab_axis
#     if slab_axis not in ("x", "y", "z"):
#         raise ValueError(f"slab_axis must be 'x', 'y', or 'z', got '{slab_axis}'")

#     # avoid GB axis = slab axis
#     if gb_axis == slab_axis:
#         remaining = [ax for ax in ("x", "y", "z") if ax != slab_axis]
#         gb_axis = remaining[0]
#         logger.warning(
#             "STEP3-SINGLE: GB_AXIS matched SLAB_AXIS. Using GB_AXIS='%s' for %s.", gb_axis, slab_gro.name
#         )

#     margin_axis = next(ax for ax in ("x", "y", "z") if ax not in (gb_axis, slab_axis))

#     # universe / coordinates
#     u = step2_out["u"]
#     residues = list(u.residues)

#     resid_to_idx: Dict[int, int] = {r.resid: i for i, r in enumerate(residues)}
#     idx_to_resid: Dict[int, int] = {i: r.resid for i, r in enumerate(residues)}

#     COM = np.array([r.atoms.center_of_mass() for r in residues])  # Å
#     coords = axis_vals_dict(COM)

#     lx, ly, lz, *_ = u.dimensions
#     box_len = {"x": float(lx), "y": float(ly), "z": float(lz)}
#     box_mid = {ax: 0.5 * box_len[ax] for ax in ("x", "y", "z")}

#     margin_mask_axis = margin_mask_1d(coords[margin_axis], box_margin)
#     slab_center = midlayer_center(coords[slab_axis])
#     layer_mask_this = layer_mask(coords[slab_axis], slab_center, slab_thick, box_len[slab_axis])

#     logger.info(
#         "[step3-single] %s: SLAB_AXIS=%s, GB_AXIS=%s, MARGIN_AXIS=%s, center=%.2f Å",
#         slab_gro.name,
#         slab_axis,
#         gb_axis,
#         margin_axis,
#         slab_center,
#     )

#     paths = step2_out["paths"]
#     G1_resids = _valid_resids(paths["g1"], resid_to_idx)
#     G2_resids = _valid_resids(paths["g2"], resid_to_idx)
#     GB_resids = _valid_resids(paths["gb"], resid_to_idx)

#     g1_idx_all = np.array([resid_to_idx[r] for r in G1_resids], dtype=int)
#     g2_idx_all = np.array([resid_to_idx[r] for r in G2_resids], dtype=int)
#     gb_idx_all = np.array([resid_to_idx[r] for r in GB_resids], dtype=int)

#     # segment GB within this layer
#     gb_idx_layer = gb_idx_all[layer_mask_this[gb_idx_all] & margin_mask_axis[gb_idx_all]]

#     if gb_idx_layer.size:
#         nbrs = NearestNeighbors(radius=connect_radius_layer).fit(COM[gb_idx_layer])
#         G = nbrs.radius_neighbors_graph(COM[gb_idx_layer], mode="connectivity")
#         seen = np.zeros(len(gb_idx_layer), dtype=bool)
#         comps: List[np.ndarray] = []

#         for s0 in range(len(gb_idx_layer)):
#             if seen[s0]:
#                 continue
#             q: deque[int] = deque([s0])
#             seen[s0] = True
#             cur: List[int] = [s0]
#             while q:
#                 i = q.popleft()
#                 for j in G[i].indices:
#                     if not seen[j]:
#                         seen[j] = True
#                         q.append(j)
#                         cur.append(j)
#             comps.append(np.array(cur, dtype=int))

#         gb_segments = [gb_idx_layer[c] for c in comps]
#     else:
#         gb_segments = []

#     slab_stem = slab_gro.stem
#     rows: List[dict] = []
#     seg_id = 0

#     axis_vals = coords[gb_axis]

#     for seg in sorted(gb_segments, key=len, reverse=True):
#         seg_id += 1
#         y0 = float(np.median(axis_vals[seg]))

#         gb_band_mask = (
#             (np.abs(axis_vals - y0) <= gb_band_thick / 2)
#             & layer_mask_this
#             & margin_mask_axis
#         )
#         gb_band_ids = np.intersect1d(seg, np.where(gb_band_mask)[0])

#         lo1, hi1 = y0 - (gb_offset + gb_band_thick), y0 - gb_offset
#         lo2, hi2 = y0 + gb_offset, y0 + (gb_offset + gb_band_thick)

#         def in_bands(arr: np.ndarray) -> np.ndarray:
#             return ((arr >= lo1) & (arr <= hi1)) | ((arr >= lo2) & (arr <= hi2))

#         g1_cand = g1_idx_all[layer_mask_this[g1_idx_all] & margin_mask_axis[g1_idx_all]]
#         g1_cand = g1_cand[in_bands(coords[gb_axis][g1_cand])]
#         if gb_idx_all.size:
#             g1_cand = np.setdiff1d(g1_cand, gb_idx_all, assume_unique=False)

#         g2_cand = g2_idx_all[layer_mask_this[g2_idx_all] & margin_mask_axis[g2_idx_all]]
#         g2_cand = g2_cand[in_bands(coords[gb_axis][g2_cand])]
#         if gb_idx_all.size:
#             g2_cand = np.setdiff1d(g2_cand, gb_idx_all, assume_unique=False)

#         g1_cand = largest_component_within(g1_cand, COM, connect_radius_layer)
#         g2_cand = largest_component_within(g2_cand, COM, connect_radius_layer)

#         if target_per_side:
#             g1_cand = cap_component(g1_cand, target_per_side, COM, connect_radius_layer)
#             g2_cand = cap_component(g2_cand, target_per_side, COM, connect_radius_layer)

#         gb_res = sorted(idx_to_resid[i] for i in gb_band_ids)
#         g1_res = sorted(idx_to_resid[i] for i in g1_cand)
#         g2_res = sorted(idx_to_resid[i] for i in g2_cand)

#         gb_n, g1_n, g2_n = len(gb_res), len(g1_res), len(g2_res)
#         dist_center = abs(y0 - box_mid[gb_axis])

#         write_paths: Optional[Dict[str, str]] = None
#         if (gb_n >= min_gb_to_write) and write_gro:
#             stem = f"{slab_stem}_seg{seg_id:02d}_{gb_axis}{y0:.2f}"
#             gb_gro = slab_dir / f"{stem}_gb.gro"
#             g1_gro = slab_dir / f"{stem}_g1.gro"
#             g2_gro = slab_dir / f"{stem}_g2.gro"

#             resid_to_idx_cache = {r.resid: i for i, r in enumerate(residues)}
#             ok1 = write_gro_for_resids(u, gb_res, str(gb_gro), resid_to_idx_cache)
#             ok2 = write_gro_for_resids(u, g1_res, str(g1_gro), resid_to_idx_cache)
#             ok3 = write_gro_for_resids(u, g2_res, str(g2_gro), resid_to_idx_cache)

#             if ok1 and ok2 and ok3:
#                 write_paths = {"gb": str(gb_gro), "g1": str(g1_gro), "g2": str(g2_gro)}
#                 if write_txt:
#                     np.savetxt(slab_dir / f"{stem}_gb.txt", np.array(gb_res, int), fmt="%d")
#                     np.savetxt(slab_dir / f"{stem}_g1.txt", np.array(g1_res, int), fmt="%d")
#                     np.savetxt(slab_dir / f"{stem}_g2.txt", np.array(g2_res, int), fmt="%d")

#         rows.append(
#             {
#                 "slab_dir": slab_dir.name,
#                 "slab_stem": slab_stem,
#                 "group_id": seg_id,
#                 "gb_axis": gb_axis,
#                 "y_center_A": y0,
#                 "dist_to_box_center_A": dist_center,
#                 "GB_N": gb_n,
#                 "G1_N": g1_n,
#                 "G2_N": g2_n,
#                 "contact_plane": "NA",
#                 "paths": write_paths,
#             }
#         )

#     return rows


# # -------------------------
# # Step 3B — 3-layer merged grain/GB selection
# # -------------------------


# def step3_merge_layers_for_slab(
#     slab_gro: str | Path,
#     slab_dir: str | Path,
#     step2_out: Dict[str, object],
#     gb_axis: str,
#     a_len: float,
#     b_len: float,
#     c_len: float,
#     alpha: float,
#     beta: float,
#     gamma: float,
#     slab_thick: float,
#     gb_offset: float,
#     gb_band_thick: float,
#     box_margin: float,
#     connect_radius_layer: float,
#     merge_tol_y: float,
#     write_txt: bool,
#     write_gro: bool,
#     min_count_write: int,
#     slab_axis: str,
# ) -> List[dict]:
#     """
#     Build and merge segments across three adjacent layers (k=-1,0,+1) around midlayer.
#     Returns the list of merged group records in the same format used by main.py.
#     """
#     slab_gro = Path(slab_gro)
#     slab_dir = Path(slab_dir)

#     # Validate slab_axis
#     if slab_axis not in ("x", "y", "z"):
#         raise ValueError(f"slab_axis must be 'x', 'y', or 'z', got '{slab_axis}'")

#     if gb_axis == slab_axis:
#         remaining = [ax for ax in ("x", "y", "z") if ax != slab_axis]
#         gb_axis = remaining[0]
#         logger.warning(
#             "STEP3: GB_AXIS matched SLAB_AXIS. Using GB_AXIS='%s' for %s.", gb_axis, slab_gro.name
#         )

#     margin_axis = next(ax for ax in ("x", "y", "z") if ax not in (gb_axis, slab_axis))
#     delta = unitcell_delta_for_axis(slab_axis, a_len, b_len, c_len)

#     u = step2_out["u"]
#     residues = list(u.residues)

#     resid_to_idx: Dict[int, int] = {r.resid: i for i, r in enumerate(residues)}
#     idx_to_resid: Dict[int, int] = {i: r.resid for i, r in enumerate(residues)}

#     COM = np.array([r.atoms.center_of_mass() for r in residues])  # Å
#     coords = axis_vals_dict(COM)

#     lx, ly, lz, *_ = u.dimensions
#     box_len = {"x": float(lx), "y": float(ly), "z": float(lz)}
#     box_mid = {ax: 0.5 * box_len[ax] for ax in ("x", "y", "z")}

#     margin_mask_axis = margin_mask_1d(coords[margin_axis], box_margin)
#     slab_center_0 = midlayer_center(coords[slab_axis])

#     logger.info(
#         "[step3] %s: SLAB_AXIS=%s, GB_AXIS=%s, MARGIN_AXIS=%s, center=%.2f Å, Δ=%.3f Å",
#         slab_gro.name,
#         slab_axis,
#         gb_axis,
#         margin_axis,
#         slab_center_0,
#         delta,
#     )

#     layer_centers = [slab_center_0 + k * delta for k in (-1, 0, +1)]

#     paths = step2_out["paths"]
#     G1_resids = _valid_resids(paths["g1"], resid_to_idx)
#     G2_resids = _valid_resids(paths["g2"], resid_to_idx)
#     GB_resids = _valid_resids(paths["gb"], resid_to_idx)

#     g1_idx_all = np.array([resid_to_idx[r] for r in G1_resids], dtype=int)
#     g2_idx_all = np.array([resid_to_idx[r] for r in G2_resids], dtype=int)
#     gb_idx_all = np.array([resid_to_idx[r] for r in GB_resids], dtype=int)

#     layer_seg_records: List[dict] = []

#     axis_vals = coords[gb_axis]

#     for layer_center in layer_centers:
#         layer_mask_this = layer_mask(coords[slab_axis], layer_center, slab_thick, box_len[slab_axis])

#         gb_idx_layer = gb_idx_all[layer_mask_this[gb_idx_all] & margin_mask_axis[gb_idx_all]]

#         if gb_idx_layer.size:
#             nbrs = NearestNeighbors(radius=connect_radius_layer).fit(COM[gb_idx_layer])
#             G = nbrs.radius_neighbors_graph(COM[gb_idx_layer], mode="connectivity")
#             seen = np.zeros(len(gb_idx_layer), dtype=bool)
#             comps: List[np.ndarray] = []

#             for s0 in range(len(gb_idx_layer)):
#                 if seen[s0]:
#                     continue
#                 q: deque[int] = deque([s0])
#                 seen[s0] = True
#                 cur: List[int] = [s0]
#                 while q:
#                     i = q.popleft()
#                     for j in G[i].indices:
#                         if not seen[j]:
#                             seen[j] = True
#                             q.append(j)
#                             cur.append(j)
#                 comps.append(np.array(cur, dtype=int))

#             gb_segments = [gb_idx_layer[c] for c in comps]
#         else:
#             gb_segments = []

#         for seg in sorted(gb_segments, key=len, reverse=True):
#             y0 = float(np.median(axis_vals[seg]))

#             gb_band_mask = (
#                 (np.abs(axis_vals - y0) <= gb_band_thick / 2)
#                 & layer_mask_this
#                 & margin_mask_axis
#             )
#             gb_band_ids = np.intersect1d(seg, np.where(gb_band_mask)[0])

#             lo1, hi1 = y0 - (gb_offset + gb_band_thick), y0 - gb_offset
#             lo2, hi2 = y0 + gb_offset, y0 + (gb_offset + gb_band_thick)

#             def in_bands(arr: np.ndarray) -> np.ndarray:
#                 return ((arr >= lo1) & (arr <= hi1)) | ((arr >= lo2) & (arr <= hi2))

#             g1_cand = g1_idx_all[layer_mask_this[g1_idx_all] & margin_mask_axis[g1_idx_all]]
#             g1_cand = g1_cand[in_bands(coords[gb_axis][g1_cand])]
#             if gb_idx_all.size:
#                 g1_cand = np.setdiff1d(g1_cand, gb_idx_all, assume_unique=False)

#             g2_cand = g2_idx_all[layer_mask_this[g2_idx_all] & margin_mask_axis[g2_idx_all]]
#             g2_cand = g2_cand[in_bands(coords[gb_axis][g2_cand])]
#             if gb_idx_all.size:
#                 g2_cand = np.setdiff1d(g2_cand, gb_idx_all, assume_unique=False)

#             layer_seg_records.append(
#                 {
#                     "y0": y0,
#                     "gb_resids": set(idx_to_resid[i] for i in gb_band_ids),
#                     "g1_resids": set(idx_to_resid[i] for i in g1_cand),
#                     "g2_resids": set(idx_to_resid[i] for i in g2_cand),
#                 }
#             )

#     # merge segments across layers
#     layer_seg_records.sort(key=lambda r: r["y0"])
#     merged_groups: List[dict] = []

#     for rec in layer_seg_records:
#         assigned = False
#         for grp in merged_groups:
#             if abs(rec["y0"] - grp["y_ref"]) <= merge_tol_y:
#                 grp["members"].append(rec)
#                 grp["y_ref"] = float(np.mean([m["y0"] for m in grp["members"]]))
#                 grp["gb_resids"].update(rec["gb_resids"])
#                 grp["g1_resids"].update(rec["g1_resids"])
#                 grp["g2_resids"].update(rec["g2_resids"])
#                 assigned = True
#                 break

#         if not assigned:
#             merged_groups.append(
#                 {
#                     "y_ref": rec["y0"],
#                     "members": [rec],
#                     "gb_resids": set(rec["gb_resids"]),
#                     "g1_resids": set(rec["g1_resids"]),
#                     "g2_resids": set(rec["g2_resids"]),
#                 }
#             )

#     # write outputs + assemble rows
#     rows: List[dict] = []
#     slab_stem = slab_gro.stem
#     resid_to_idx_cache = {r.resid: i for i, r in enumerate(residues)}

#     for gid, grp in enumerate(merged_groups, start=1):
#         yref = grp["y_ref"]
#         gb_res = sorted(grp["gb_resids"])
#         g1_res = sorted(grp["g1_resids"])
#         g2_res = sorted(grp["g2_resids"])

#         gb_n, g1_n, g2_n = len(gb_res), len(g1_res), len(g2_res)
#         dist_center = abs(yref - box_mid[gb_axis])

#         write_paths: Optional[Dict[str, str]] = None
#         if gb_n >= min_count_write and g1_n >= min_count_write and g2_n >= min_count_write and write_gro:
#             stem = f"{slab_stem}_grp{gid:02d}_{gb_axis}{yref:0.2f}"
#             gb_gro = slab_dir / f"{stem}_gb.gro"
#             g1_gro = slab_dir / f"{stem}_g1.gro"
#             g2_gro = slab_dir / f"{stem}_g2.gro"

#             ok1 = write_gro_for_resids(u, gb_res, str(gb_gro), resid_to_idx_cache)
#             ok2 = write_gro_for_resids(u, g1_res, str(g1_gro), resid_to_idx_cache)
#             ok3 = write_gro_for_resids(u, g2_res, str(g2_gro), resid_to_idx_cache)

#             if ok1 and ok2 and ok3:
#                 write_paths = {"gb": str(gb_gro), "g1": str(g1_gro), "g2": str(g2_gro)}
#                 if write_txt:
#                     np.savetxt(slab_dir / f"{stem}_gb.txt", np.array(gb_res, int), fmt="%d")
#                     np.savetxt(slab_dir / f"{stem}_g1.txt", np.array(g1_res, int), fmt="%d")
#                     np.savetxt(slab_dir / f"{stem}_g2.txt", np.array(g2_res, int), fmt="%d")

#         rows.append(
#             {
#                 "slab_dir": slab_dir.name,
#                 "slab_stem": slab_stem,
#                 "group_id": gid,
#                 "gb_axis": gb_axis,
#                 "y_center_A": yref,
#                 "dist_to_box_center_A": dist_center,
#                 "GB_N": gb_n,
#                 "G1_N": g1_n,
#                 "G2_N": g2_n,
#                 "contact_plane": "NA",
#                 "paths": write_paths,
#             }
#         )

#     return rows


