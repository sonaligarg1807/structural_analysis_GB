# step2.py
from __future__ import annotations

from pathlib import Path
from collections import deque
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


Label = Literal["grain1", "grain2", "GB"]


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
) -> Dict[str, object]:
    """
    Step 2 — classify slab residues into grain1 / GB / grain2 (topological).

    Workflow (per slab):
      1. Load `slab_gro` into a Universe.
      2. For each residue:
         - Compute COM
         - Compute gyration long axis (orientation)
         - Map orientation to a 6D nematic embedding.
      3. Cluster orientations with KMeans in nematic-embedded space → two
         grain directions g1, g2 (headless).
      4. Initial labeling based on alignment with g1 / g2:
         - 'grain1' if strongly aligned with g1
         - 'grain2' if strongly aligned with g2
         - 'GB' otherwise.
      5. Spatial smoothing of labels using PBC neighbors.
      6. Identify boundary residues with neighbors in the opposite grain,
         then dilate this set to obtain a GB band.
      7. Drop tiny GB connected components (< min_gb_size) by assigning
         them to the dominant neighboring grain.
      8. Final labels: everything in GB_mask → 'GB'; rest keep smoothed label.
      9. Write three TXT files with resids for grain1 / GB / grain2.

    Parameters
    ----------
    slab_gro
        Path to the slab .gro file produced by step1.
    slab_dir
        Directory where the TXT output files will be written.
    out_prefix
        Prefix for the TXT filenames, e.g. 'topo' → topo_<stem>_grain1.txt.
    th_high
        Cosine threshold: require cos(theta) >= th_high to assign to a grain.
    margin
        Margin threshold: require |cos1 - cos2| >= margin to assign.
    smooth_iters
        Number of label-smoothing iterations using neighbor voting.
    connect_radius
        Radius (Angstrom) for PBC neighbor graph used in smoothing / components.
    edge_radius
        Radius (Angstrom) for detecting “opposite-grain” neighbors at the GB.
    opp_min_nb
        Minimum count of opposite-grain neighbors to mark a boundary site.
    edge_dilate_steps
        Number of dilation steps to grow the GB band.
    min_gb_size
        Minimum allowed GB component size. Smaller connected components
        are re-assigned to the dominant neighboring grain.

    Returns
    -------
    dict
        {
          "paths": {"g1": Path, "gb": Path, "g2": Path},
          "u": Universe rooted at slab_gro (fresh instance),
        }
    """
    slab_gro = Path(slab_gro)
    slab_dir = Path(slab_dir)

    # ------------------------------------------------------------------
    # Load slab and compute basic per-residue features
    # ------------------------------------------------------------------
    u = mda.Universe(str(slab_gro))
    residues = list(u.residues)
    N = len(residues)

    resids = np.array([r.resid for r in residues], dtype=int)
    COM = np.array([r.atoms.center_of_mass() for r in residues])
    dirs = np.array([gyration_long_axis(r.atoms) for r in residues])  # unit vectors
    feat6 = np.array([nematic_embed(d) for d in dirs])  # 6D headless embedding
    box = u.dimensions

    # ------------------------------------------------------------------
    # 1) KMeans in nematic-embedded space → two grain directions g1, g2
    # ------------------------------------------------------------------
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(feat6)
    g1 = nematic_center_to_director(km.cluster_centers_[0])
    g2 = nematic_center_to_director(km.cluster_centers_[1])

    # deterministic ordering of grain directions
    if tuple(np.round(g2, 6)) < tuple(np.round(g1, 6)):
        g1, g2 = g2, g1

    # absolute cosines for headless alignment
    cos1 = np.abs(dirs @ g1)
    cos2 = np.abs(dirs @ g2)
    maxcos = np.maximum(cos1, cos2)
    diff = np.abs(cos1 - cos2)

    # ------------------------------------------------------------------
    # 2) Initial labeling: grain1 / grain2 / GB
    # ------------------------------------------------------------------
    labels: np.ndarray = np.array(["GB"] * N, dtype=object)

    labels[(cos1 >= th_high) & (cos1 - cos2 >= margin)] = "grain1"
    labels[(cos2 >= th_high) & (cos2 - cos1 >= margin)] = "grain2"

    # neighbor lists (indices) under PBC
    neigh = pbc_radius_neighbors(COM, connect_radius, box)
    neigh_edge = pbc_radius_neighbors(COM, edge_radius, box)

    # ------------------------------------------------------------------
    # 3) Spatial smoothing using neighbor voting
    # ------------------------------------------------------------------
    def smooth_once(lbls: np.ndarray) -> np.ndarray:
        new = lbls.copy()
        for i, nb in enumerate(neigh):
            if nb.size == 0:
                continue

            counts = {"grain1": 0, "grain2": 0, "GB": 0}
            for j in nb:
                counts[lbls[j]] += 1

            dom = max(counts, key=counts.get)
            tot = sum(counts.values())

            # only adopt a dominant grain label if:
            #  (a) it's not GB and
            #  (b) it has ≥ 60% of neighbors and
            #  (c) local orientation is not clearly ambiguous
            if dom != "GB" and counts[dom] >= 0.6 * max(tot, 1):
                if not ((maxcos[i] < th_high) or (diff[i] < margin)):
                    new[i] = dom
        return new

    for _ in range(max(0, smooth_iters)):
        labels = smooth_once(labels)

    # sets for quick membership checks
    g1_idx = np.where(labels == "grain1")[0]
    g2_idx = np.where(labels == "grain2")[0]
    g1_set, g2_set = set(g1_idx), set(g2_idx)

    # ------------------------------------------------------------------
    # 4) Identify boundary sites (potential GB) via opposite-grain neighbors
    # ------------------------------------------------------------------
    is_boundary = np.zeros(N, dtype=bool)
    for i in range(N):
        nb = neigh_edge[i]
        if labels[i] == "grain1":
            if sum((j in g2_set) for j in nb) >= opp_min_nb:
                is_boundary[i] = True
        elif labels[i] == "grain2":
            if sum((j in g1_set) for j in nb) >= opp_min_nb:
                is_boundary[i] = True
        else:
            # label is already GB → considered boundary
            is_boundary[i] = True

    # ------------------------------------------------------------------
    # 5) Dilation: expand boundary into a GB band
    # ------------------------------------------------------------------
    GB_mask = is_boundary.copy()
    for _ in range(max(0, edge_dilate_steps)):
        expand = GB_mask.copy()
        for i in np.where(GB_mask)[0]:
            for j in neigh[i]:
                expand[j] = True
        GB_mask = expand

    # ------------------------------------------------------------------
    # 6) Drop tiny GB components by reassigning them to neighbor grains
    # ------------------------------------------------------------------
    if min_gb_size > 0:
        visited = np.zeros(N, dtype=bool)

        for i in range(N):
            if visited[i] or not GB_mask[i]:
                continue

            # BFS to gather one connected component of GB_mask
            comp: list[int] = []
            q: deque[int] = deque([i])
            visited[i] = True

            while q:
                k = q.popleft()
                comp.append(k)
                for j in neigh[k]:
                    if GB_mask[j] and not visited[j]:
                        visited[j] = True
                        q.append(j)

            # reassign if component is too small
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

    # ------------------------------------------------------------------
    # 7) Final labels: override GB band
    # ------------------------------------------------------------------
    final = labels.copy()
    final[GB_mask] = "GB"

    # ------------------------------------------------------------------
    # 8) Write outputs: TXT files for grain1 / GB / grain2 resids
    # ------------------------------------------------------------------
    out_g1 = slab_dir / f"{out_prefix}_{slab_gro.stem}_grain1.txt"
    out_gb = slab_dir / f"{out_prefix}_{slab_gro.stem}_GB.txt"
    out_g2 = slab_dir / f"{out_prefix}_{slab_gro.stem}_grain2.txt"

    np.savetxt(out_g1, resids[final == "grain1"], fmt="%d")
    np.savetxt(out_gb, resids[final == "GB"], fmt="%d")
    np.savetxt(out_g2, resids[final == "grain2"], fmt="%d")

    n_g1 = int(np.sum(final == "grain1"))
    n_gb = int(np.sum(final == "GB"))
    n_g2 = int(np.sum(final == "grain2"))

    print(
        f"[step2] {slab_gro.name}: "
        f"grain1={n_g1}  GB={n_gb}  grain2={n_g2}"
    )

    # fresh Universe, as in your original code
    return {
        "paths": {"g1": out_g1, "gb": out_gb, "g2": out_g2},
        "u": mda.Universe(str(slab_gro)),
    }
