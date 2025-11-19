# step1.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.GRO import GROWriter
from sklearn.cluster import DBSCAN

from src.utils import (
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
    in_gro: str | Path,
    resname: str,
    out_dir: str | Path,
    out_base: str,
    thickness_A: float,
    neigh_cutoff: float,
    smooth_iters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    select_by: Literal["resid", "atom"],
    unwrap: bool,
    write_summary: bool = True,
) -> list[Path]:
    """
    Step 1 — extract GB slabs from a full bicrystal system.

    Workflow:
      1. Load full system from `in_gro` via MDAnalysis.
      2. Compute COM + principal normals for all residues with `resname`.
      3. Split orientations into two grains (spherical k-means, headless).
      4. Detect GB molecules via local label mixing (PBC-aware).
      5. Cluster GB COMs with DBSCAN to identify distinct GB segments.
      6. For each GB cluster:
         - Fit a plane.
         - Select atoms inside a slab of thickness ±thickness_A/2 along the
           plane normal.
         - Write a slab .gro and metadata .json into per-slab subdirectories.
      7. Optionally, write a plain-text summary file in `out_dir`.

    Parameters
    ----------
    in_gro
        Path to the input bicrystal .gro file.
    resname
        Residue name used to identify the molecular species of interest
        (e.g. 'EN-' for pentacene).
    out_dir
        Directory in which per-slab subdirectories will be created.
    out_base
        Base stem used when naming outputs, e.g. 'gb_slab'.
    thickness_A
        Slab half-thickness in Angstrom; actual window is ± thickness_A/2
        along the GB-plane normal.
    neigh_cutoff
        Cutoff distance (in Angstrom) for spatial smoothing / GB detection.
    smooth_iters
        Number of smoothing iterations applied to grain labels.
    dbscan_eps
        DBSCAN epsilon (distance in Angstrom) for GB cluster detection.
    dbscan_min_samples
        DBSCAN min_samples parameter.
    select_by
        'resid' → select residues by COM distance from the GB plane
        'atom'  → select atoms by distance from the GB plane.
    unwrap
        If True, apply MDAnalysis `unwrap` to remove PBC jumps.
    write_summary
        If True, write `summary_step1_slabs.txt` in `out_dir`.

    Returns
    -------
    written
        List of paths to written slab .gro files.
    """
    in_gro = str(in_gro)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "extracted_slabs_summary.txt"
    logs: list[str] = []

    print(f"[step1] Loading full system from: {in_gro}")
    u = mda.Universe(in_gro)

    if unwrap:
        from MDAnalysis.transformations import unwrap as _unwrap

        u.trajectory.add_transformations(_unwrap(u.atoms))

    ts = u.trajectory.ts
    a_vec, b_vec, c_vec = get_cell_vectors_A(ts)

    # ------------------------------------------------------------------
    # collect COM + normals per residue for the target resname
    # ------------------------------------------------------------------
    pen_res = [r for r in u.residues if r.resname == resname]
    if not pen_res:
        raise RuntimeError(f"No residues named {resname} in {in_gro}")

    coms: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    resids: list[int] = []

    heavy_all = u.select_atoms("not name H*")

    for r in pen_res:
        ag = r.atoms
        # prioritize heavy atoms; fall back to all atoms if necessary
        pts = (ag & heavy_all).positions
        if len(pts) < 6:
            pts = ag.select_atoms("not name H*").positions
        if len(pts) < 6:
            # skip very small or malformed residues
            continue

        coms.append(ag.center_of_mass())
        normals.append(principal_normal(pts))
        resids.append(r.resid)

    coms = np.asarray(coms)
    normals = np.asarray(normals)
    resids = np.asarray(resids, dtype=int)

    print(f"[step1] Molecules used for orientation/GB detection: {len(coms)}")

    # ------------------------------------------------------------------
    # grain labels, smoothing, GB detection
    # ------------------------------------------------------------------
    labels = spherical_2means_headless(normals, iters=20, seed=0)
    labels = smooth_labels_spatial(
        coms,
        labels,
        box=u.dimensions,
        cutoff=neigh_cutoff,
        iters=smooth_iters,
    )

    is_gb = detect_gb_pbc(coms, labels, box=u.dimensions, cutoff=neigh_cutoff)
    gb_idx = np.where(is_gb)[0]
    if gb_idx.size == 0:
        raise RuntimeError("[step1] No GB contacts found for this system.")

    gb_pts = coms[gb_idx]

    # ------------------------------------------------------------------
    # cluster GB molecules into distinct GB segments
    # ------------------------------------------------------------------
    gb_labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(
        gb_pts
    )

    clusters: list[tuple[int, np.ndarray]] = []
    for lab in sorted(set(gb_labels)):
        if lab < 0:
            continue  # DBSCAN noise
        member_idx = gb_idx[gb_labels == lab]
        clusters.append((lab, member_idx))

    # sort by cluster size (largest first)
    clusters.sort(key=lambda x: len(x[1]), reverse=True)

    half = 0.5 * thickness_A
    written: list[Path] = []

    print(f"[step1] Found {len(clusters)} GB clusters from DBSCAN.")

    for rank, (lab, member_idx) in enumerate(clusters, start=1):
        pts = coms[member_idx]
        ctr, n = fit_plane(pts)

        cart_k, cart_ang, lat_k, lat_ang = axis_report(n, a_vec, b_vec, c_vec)

        # --------------------------------------------------------------
        # select slab atoms either by residue or atom distances
        # --------------------------------------------------------------
        d_res = signed_distance_along_vec(coms, ctr, n)

        if select_by.lower() == "resid":
            keep_mask = (d_res >= -half) & (d_res <= half)
            keep_resids = resids[keep_mask].tolist()

            if keep_resids:
                sel = "resid " + " ".join(map(str, keep_resids))
                ag = u.select_atoms(sel)
            else:
                ag = u.atoms[:0]
        else:
            # select by atom distances
            d_atom = signed_distance_along_vec(u.atoms.positions, ctr, n)
            keep_atoms = (d_atom >= -half) & (d_atom <= half)
            ag = u.atoms[keep_atoms]

        # --------------------------------------------------------------
        # write slab .gro + metadata JSON
        # --------------------------------------------------------------
        # slab_stem = (
        #     f"{out_base}_slab_rank{rank:02d}_N{len(member_idx)}_th{int(thickness_A)}A"
        # )
        slab_stem = (f"slab_{rank:02d}")
        slab_dir = out_dir / slab_stem
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
            f"cart={cart_k}({cart_ang:.2f}°)\t"
            f"latt={lat_k}({lat_ang:.2f}°)"
        )
        written.append(out_gro)

        print(f"[step1] Wrote slab #{rank:02d} → {out_gro}")

    # ------------------------------------------------------------------
    # summary file
    # ------------------------------------------------------------------
    if write_summary:
        header = "rank\tlabel\tsize\tdir\tfile\tbest_cart\tbest_latt\n"
        summary = header + "\n".join(logs) + "\n"
        summary_path.write_text(summary)
        print(
            f"[step1] Done. Slabs written: {len(written)} ; "
            f"summary: {summary_path}"
        )
    else:
        print(f"[step1] Done. Slabs written: {len(written)} ; summary disabled.")

    return written
