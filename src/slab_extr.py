from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Tuple

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

logger = logging.getLogger(__name__)

__all__ = [
    "step1_extract_all_slabs",
    "compute_residue_orientations",
    "detect_and_cluster_gb_pts",
    "extract_slab_atoms",
]


def compute_residue_orientations(universe: mda.Universe, resname: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-residue COMs, principal normals and residue ids for residues matching `resname`.

    Returns
    -------
    coms, normals, resids
    """
    pen_res = [r for r in universe.residues if r.resname == resname]
    if not pen_res:
        raise RuntimeError(f"No residues named {resname} in provided Universe")

    heavy_all = universe.select_atoms("not name H*")

    coms = []
    normals = []
    resids = []
    for r in pen_res:
        ag = r.atoms
        pts = (ag & heavy_all).positions
        if len(pts) < 6:
            pts = ag.select_atoms("not name H*").positions
        if len(pts) < 6:
            # skip very small or malformed residues
            continue
        coms.append(ag.center_of_mass())
        normals.append(principal_normal(pts))
        resids.append(r.resid)

    return np.asarray(coms), np.asarray(normals), np.asarray(resids, dtype=int)


def detect_and_cluster_gb_pts(
    coms: np.ndarray,
    normals: np.ndarray,
    resids: np.ndarray,
    box: Iterable[float],
    neigh_cutoff: float,
    smooth_iters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
):
    """
    Given per-residue COMs and normals, compute smoothed grain labels, detect GB residues
    and cluster them using DBSCAN. Returns a sorted list of clusters [(label, member_idx), ...].
    """
    labels = spherical_2means_headless(normals, iters=20, seed=0)
    labels = smooth_labels_spatial(
        coms, labels, box=box, cutoff=neigh_cutoff, iters=smooth_iters
    )

    is_gb = detect_gb_pbc(coms, labels, box=box, cutoff=neigh_cutoff)
    gb_idx = np.where(is_gb)[0]
    if gb_idx.size == 0:
        raise RuntimeError("[step1] No GB contacts found for this system.")

    gb_pts = coms[gb_idx]
    gb_labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(
        gb_pts
    )

    clusters = []
    for lab in sorted(set(gb_labels)):
        if lab < 0:
            continue  # noise
        member_idx = gb_idx[gb_labels == lab]
        clusters.append((lab, member_idx))

    clusters.sort(key=lambda x: len(x[1]), reverse=True)
    return clusters


def extract_slab_atoms(
    universe: mda.Universe,
    ctr: np.ndarray,
    n: np.ndarray,
    thickness_A: float,
    select_by: str = "resid",
):
    """
    Select and return an MDAnalysis AtomGroup corresponding to the slab (±thickness_A/2 along normal).

    Parameters
    ----------
    universe: MDAnalysis.Universe
    ctr: center point of the fitted plane
    n: plane normal
    thickness_A: slab thickness in Angstroms
    select_by: 'resid' to select residues by COM, otherwise selects by atom distances
    """
    half = 0.5 * thickness_A
    if select_by.lower() == "resid":
        res_coms = np.array([r.center_of_mass() for r in universe.residues])
        d_res = signed_distance_along_vec(res_coms, ctr, n)
        keep_mask = (d_res >= -half) & (d_res <= half)
        keep_residues = [r.resid for r, keep in zip(universe.residues, keep_mask) if keep]
        if keep_residues:
            sel = "resid " + " ".join(map(str, keep_residues))
            ag = universe.select_atoms(sel)
        else:
            ag = universe.atoms[:0]
    else:
        d_atom = signed_distance_along_vec(universe.atoms.positions, ctr, n)
        keep_atoms = (d_atom >= -half) & (d_atom <= half)
        ag = universe.atoms[keep_atoms]

    return ag


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
    select_by: str,
    unwrap: bool,
    write_summary: bool = True,
) -> list[Path]:
    """
    Orchestrate Step 1: load system, determine residue orientations, detect GB regions,
    cluster GB residues and write per-slab GRO + metadata JSON files.

    Returns
    -------
    List[Path] of written slab .gro files.
    """
    in_gro = str(in_gro)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "extracted_slabs_summary.txt"
    logs: list[str] = []

    logger.info("[step1] Loading full system from: %s", in_gro)
    u = mda.Universe(in_gro)

    if unwrap:
        from MDAnalysis.transformations import unwrap as _unwrap

        u.trajectory.add_transformations(_unwrap(u.atoms))

    ts = u.trajectory.ts
    a_vec, b_vec, c_vec = get_cell_vectors_A(ts)

    coms, normals, resids = compute_residue_orientations(u, resname)
    logger.info("[step1] Molecules used for orientation/GB detection: %d", len(coms))

    clusters = detect_and_cluster_gb_pts(
        coms=coms,
        normals=normals,
        resids=resids,
        box=u.dimensions,
        neigh_cutoff=neigh_cutoff,
        smooth_iters=smooth_iters,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )

    written: list[Path] = []
    logger.info("[step1] Found %d GB clusters from DBSCAN.", len(clusters))

    for rank, (lab, member_idx) in enumerate(clusters, start=1):
        pts = coms[member_idx]
        ctr, n = fit_plane(pts)

        cart_k, cart_ang, lat_k, lat_ang = axis_report(n, a_vec, b_vec, c_vec)

        # select slab atoms either by residue or atom distances
        if select_by.lower() == "resid":
            d_res = signed_distance_along_vec(coms, ctr, n)
            keep_mask = (d_res >= -0.5 * thickness_A) & (d_res <= 0.5 * thickness_A)
            keep_resids = resids[keep_mask].tolist()
            if keep_resids:
                ag = u.select_atoms("resid " + " ".join(map(str, keep_resids)))
            else:
                ag = u.atoms[:0]
        else:
            d_atom = signed_distance_along_vec(u.atoms.positions, ctr, n)
            keep_atoms = (d_atom >= -0.5 * thickness_A) & (d_atom <= 0.5 * thickness_A)
            ag = u.atoms[keep_atoms]

        slab_stem = f"slab_{rank:02d}"
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

        logger.info("[step1] Wrote slab #%02d → %s", rank, out_gro)

    if write_summary:
        header = "rank\tlabel\tsize\tdir\tfile\tbest_cart\tbest_latt\n"
        summary = header + "\n".join(logs) + "\n"
        summary_path.write_text(summary)
        logger.info("[step1] Done. Slabs written: %d ; summary: %s", len(written), summary_path)
    else:
        logger.info("[step1] Done. Slabs written: %d ; summary disabled.", len(written))

    return written