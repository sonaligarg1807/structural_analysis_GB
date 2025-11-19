# step2.py
import re
from pathlib import Path
from collections import deque

import numpy as np
import MDAnalysis as mda
from sklearn.cluster import KMeans

from .utils import (
    pbc_radius_neighbors,
    gyration_long_axis,
    nematic_embed,
    nematic_center_to_director,
)


def step2_clustering_for_slab(
    slab_gro,
    slab_dir,
    out_prefix,
    th_high,
    margin,
    smooth_iters,
    connect_radius,
    edge_radius,
    opp_min_nb,
    edge_dilate_steps,
    min_gb_size,
):
    """
    Step 2 — classify residues into grain1 / GB / grain2 (topological).
    Writes 3 TXT files under slab_dir and returns a dict with paths and a fresh Universe.
    """
    u = mda.Universe(str(slab_gro))
    residues = list(u.residues)
    N = len(residues)
    resids = np.array([r.resid for r in residues], int)
    COM = np.array([r.atoms.center_of_mass() for r in residues])
    dirs = np.array([gyration_long_axis(r.atoms) for r in residues])
    feat6 = np.array([nematic_embed(d) for d in dirs])
    box = u.dimensions

    # KMeans in nematic-embedded space
    km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(feat6)
    g1 = nematic_center_to_director(km.cluster_centers_[0])
    g2 = nematic_center_to_director(km.cluster_centers_[1])
    if tuple(np.round(g2, 6)) < tuple(np.round(g1, 6)):
        g1, g2 = g2, g1

    cos1 = np.abs(dirs @ g1)
    cos2 = np.abs(dirs @ g2)
    maxcos = np.maximum(cos1, cos2)
    diff = np.abs(cos1 - cos2)

    labels = np.array(["GB"] * N, dtype=object)
    labels[(cos1 >= th_high) & (cos1 - cos2 >= margin)] = "grain1"
    labels[(cos2 >= th_high) & (cos2 - cos1 >= margin)] = "grain2"

    neigh = pbc_radius_neighbors(COM, connect_radius, box)
    neigh_edge = pbc_radius_neighbors(COM, edge_radius, box)

    def smooth_once(lbls):
        new = lbls.copy()
        for i, nb in enumerate(neigh):
            if nb.size == 0:
                continue
            counts = {"grain1": 0, "grain2": 0, "GB": 0}
            for j in nb:
                counts[lbls[j]] += 1
            dom = max(counts, key=counts.get)
            tot = sum(counts.values())
            if dom != "GB" and counts[dom] >= 0.6 * max(tot, 1):
                if not ((maxcos[i] < th_high) or (diff[i] < margin)):
                    new[i] = dom
        return new

    for _ in range(max(0, smooth_iters)):
        labels = smooth_once(labels)

    g1_idx = np.where(labels == "grain1")[0]
    g2_idx = np.where(labels == "grain2")[0]
    g1_set, g2_set = set(g1_idx), set(g2_idx)

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
            is_boundary[i] = True

    GB_mask = is_boundary.copy()
    for _ in range(max(0, edge_dilate_steps)):
        expand = GB_mask.copy()
        for i in np.where(GB_mask)[0]:
            for j in neigh[i]:
                expand[j] = True
        GB_mask = expand

    if min_gb_size > 0:
        visited = np.zeros(N, bool)
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

    final = labels.copy()
    final[GB_mask] = "GB"

    out1 = Path(slab_dir) / f"{out_prefix}_{slab_gro.stem}_grain1.txt"
    out2 = Path(slab_dir) / f"{out_prefix}_{slab_gro.stem}_GB.txt"
    out3 = Path(slab_dir) / f"{out_prefix}_{slab_gro.stem}_grain2.txt"
    np.savetxt(out1, resids[final == "grain1"], fmt="%d")
    np.savetxt(out2, resids[final == "GB"], fmt="%d")
    np.savetxt(out3, resids[final == "grain2"], fmt="%d")

    print(
        f"[step2] {slab_gro.name}: grain1={np.sum(final=='grain1')}  "
        f"GB={np.sum(final=='GB')}  grain2={np.sum(final=='grain2')}"
    )
    return {
        "paths": {"g1": out1, "gb": out2, "g2": out3},
        "u": mda.Universe(str(slab_gro)),  # fresh universe rooted at slab
    }
