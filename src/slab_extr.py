# step1.py
import json
import os
from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.GRO import GROWriter
from sklearn.cluster import DBSCAN

from .utils import (
    principal_normal,
    spherical_2means_headless,
    smooth_labels_spatial,
    detect_gb_pbc,
    fit_plane,
    signed_distance_along_vec,
    get_cell_vectors_A,
    axis_report,
)


def step1_extract_all_slabs(
    in_gro,
    resname,
    out_dir,
    out_base,
    thickness_A,
    neigh_cutoff,
    smooth_iters,
    dbscan_eps,
    dbscan_min_samples,
    select_by,
    unwrap,
    write_summary=True,
):
    """
    Step 1 — extract GB slabs from full system.
    Returns list[Path] of written slab .gro files.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary_path = Path(out_dir) / "summary_step1_slabs.txt"
    logs = []

    print("[step1] Loading full system…")
    u = mda.Universe(in_gro)
    if unwrap:
        from MDAnalysis.transformations import unwrap as _unwrap

        u.trajectory.add_transformations(_unwrap(u.atoms))
    ts = u.trajectory.ts
    a_vec, b_vec, c_vec = get_cell_vectors_A(ts)

    pen_res = [r for r in u.residues if r.resname == resname]
    if not pen_res:
        raise RuntimeError(f"No residues named {resname} in {in_gro}")

    coms, normals, resids = [], [], []
    heavy_all = u.select_atoms("not name H*")
    for r in pen_res:
        ag = r.atoms
        pts = (ag & heavy_all).positions
        if len(pts) < 6:
            pts = ag.select_atoms("not name H*").positions
        if len(pts) < 6:
            continue
        coms.append(ag.center_of_mass())
        normals.append(principal_normal(pts))
        resids.append(r.resid)

    coms = np.asarray(coms)
    normals = np.asarray(normals)
    resids = np.asarray(resids)
    print(f"[step1] Molecules used: {len(coms)}")

    labels = spherical_2means_headless(normals, iters=20, seed=0)
    labels = smooth_labels_spatial(coms, labels, box=u.dimensions, cutoff=neigh_cutoff, iters=smooth_iters)
    is_gb = detect_gb_pbc(coms, labels, box=u.dimensions, cutoff=neigh_cutoff)
    gb_idx = np.where(is_gb)[0]
    if len(gb_idx) == 0:
        raise RuntimeError("No GB contacts found")

    gb_pts = coms[gb_idx]
    gb_labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(gb_pts)

    clusters = []
    for lab in sorted(set(gb_labels)):
        if lab < 0:
            continue
        member_idx = gb_idx[gb_labels == lab]
        clusters.append((lab, member_idx))
    clusters.sort(key=lambda x: len(x[1]), reverse=True)

    half = 0.5 * thickness_A
    written = []
    for rank, (lab, member_idx) in enumerate(clusters, start=1):
        pts = coms[member_idx]
        ctr, n = fit_plane(pts)

        cart_k, cart_ang, lat_k, lat_ang = axis_report(n, a_vec, b_vec, c_vec)

        # select atoms in thickness window along exact normal
        d_res = signed_distance_along_vec(coms, ctr, n)
        if select_by.lower() == "resid":
            keep_mask = (d_res >= -half) & (d_res <= half)
            keep_resids = resids[keep_mask].tolist()
            sel = "resid " + " ".join(map(str, keep_resids)) if keep_resids else "none"
            ag = u.select_atoms(sel) if keep_resids else u.atoms[:0]
        else:
            d_atom = signed_distance_along_vec(u.atoms.positions, ctr, n)
            keep_atoms = (d_atom >= -half) & (d_atom <= half)
            ag = u.atoms[keep_atoms]

        slab_stem = f"{out_base}_slab_rank{rank:02d}_N{len(member_idx)}_th{int(thickness_A)}A"
        slab_dir = Path(out_dir) / slab_stem
        slab_dir.mkdir(parents=True, exist_ok=True)

        out_gro = slab_dir / f"{slab_stem}.gro"
        with GROWriter(str(out_gro), n_atoms=len(ag)) as w:
            w.write(ag)

        meta = {
            "ctr": ctr.tolist(),
            "normal": (n / np.linalg.norm(n)).tolist(),
            "cart_best_axis": cart_k,
            "cart_angle_deg": cart_ang,
            "latt_best_axis": lat_k,
            "latt_angle_deg": lat_ang,
        }
        meta_path = slab_dir / f"{slab_stem}.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        logs.append(
            f"rank={rank:02d}\tlabel={lab}\tsize={len(member_idx)}\t"
            f"dir={slab_dir.name}\tfile={out_gro.name}\t"
            f"cart={cart_k}({cart_ang:.2f}°)\tlatt={lat_k}({lat_ang:.2f}°)"
        )
        written.append(out_gro)

        print(f"[step1] Wrote slab #{rank:02d} → {out_gro}")

    if write_summary:
        summary = "rank\tlabel\tsize\tdir\tfile\tbest_cart\tbest_latt\n" + "\n".join(logs) + "\n"
        summary_path.write_text(summary)
        print(f"[step1] Done. Slabs: {len(written)} ; summary: {summary_path}")
    else:
        print(f"[step1] Done. Slabs: {len(written)} ; summary not written (disabled).")

    return written  # list[Path]
