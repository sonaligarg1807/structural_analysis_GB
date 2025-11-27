#version which could switch between PCA components for slab extraction

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


def compute_residue_orientations(
    universe: mda.Universe,
    resname: str,
    orient_mode: str = "normal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-residue COMs, orientation vectors (from mass-weighted PCA)
    and residue ids for residues matching `resname`.

    Parameters
    ----------
    universe : MDAnalysis.Universe
    resname  : str
        Residue name to use (e.g. 'PEN').
    orient_mode : str, optional
        Which PCA axis (or combination) to use as the orientation vector
        for grain detection. Allowed values (case-insensitive):
            - 'long'              : largest-variance axis
            - 'short', 'small'    : intermediate axis
            - 'normal'            : smallest-variance axis (plane normal)
            - 'short_plus_normal',
              'small_plus_normal' : sum of short + normal axes

    Returns
    -------
    coms   : (N, 3) array of residue COMs
    normals: (N, 3) array of chosen orientation vectors (unit)
    resids : (N,)   array of residue ids (int)
    """
    mode = orient_mode.lower()
    allowed = {
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

    # Filter residues by name
    pen_res = [r for r in universe.residues if r.resname == resname]
    if not pen_res:
        raise RuntimeError(f"No residues named {resname} in provided Universe")

    # Heavy-atom pool (for robustness)
    heavy_all = universe.select_atoms("not name H*")

    coms = []
    orient_vecs = []
    resids = []

    for r in pen_res:
        # heavy-atom subset of this residue
        ag = r.atoms & heavy_all
        if len(ag) < 6:
            ag = r.atoms.select_atoms("not name H*")
        if len(ag) < 6:
            # skip very small or malformed residues
            continue

        # Mass-weighted COM from MDAnalysis
        com = ag.center_of_mass()

        # Positions relative to COM
        coords = ag.positions - com  # shape (n_atoms, 3)
        if coords.shape[0] < 3:
            # Need at least 3 points for a meaningful PCA
            continue

        # Masses (may be None or uniform)
        masses = ag.masses
        if masses is None or np.allclose(masses, masses[0]):
            # No meaningful mass info → fall back to unweighted PCA
            X = coords
        else:
            m_sqrt = np.sqrt(masses).reshape(-1, 1)
            X = coords * m_sqrt  # mass-weighted coordinates

        # Covariance-like matrix and eigen-decomposition
        cov = X.T @ X  # (3,3)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            # numerical issue, skip this residue
            continue

        # Sort eigenvectors by descending eigenvalue:
        # eigenvalues: ascending by default → reverse to get largest → smallest
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        # Now:
        #   long_axis   = eigvecs[:, 0]  (largest variance)
        #   short_axis  = eigvecs[:, 1]
        #   normal_axis = eigvecs[:, 2]  (smallest variance)

        long_axis = eigvecs[:, 0]
        short_axis = eigvecs[:, 1]
        normal_axis = eigvecs[:, 2]

        # Choose which axis to use as orientation
        if mode == "long":
            v = long_axis
        elif mode in ("short", "small"):
            v = short_axis
        elif mode == "normal":
            v = normal_axis
        elif mode in ("short_plus_normal", "small_plus_normal"):
            v = short_axis + normal_axis
        else:
            # Should never hit this because of earlier check
            v = normal_axis

        # Normalize and handle degenerate cases
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            # Fallback: use normal_axis if available
            norm_n = np.linalg.norm(normal_axis)
            if norm_n < 1e-8:
                continue
            v = normal_axis / norm_n
        else:
            v = v / norm

        coms.append(com)
        orient_vecs.append(v)
        resids.append(r.resid)

    if not coms:
        raise RuntimeError(
            f"No valid residues found for resname={resname} after mwPCA orientation."
        )

    return (
        np.asarray(coms),
        np.asarray(orient_vecs),
        np.asarray(resids, dtype=int),
    )


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
    # 2-grain clustering on the chosen orientation vectors (headless)
    labels = spherical_2means_headless(normals, iters=20, seed=0)

    # Spatial smoothing of grain labels
    labels = smooth_labels_spatial(
        coms, labels, box=box, cutoff=neigh_cutoff, iters=smooth_iters
    )

    # GB detection: residues that see neighbors of the other grain
    is_gb = detect_gb_pbc(coms, labels, box=box, cutoff=neigh_cutoff)
    gb_idx = np.where(is_gb)[0]
    if gb_idx.size == 0:
        raise RuntimeError("[step1] No GB contacts found for this system.")

    # Cluster GB COMs to identify distinct GB patches
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

    # Sort clusters by size (largest GB first)
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
        keep_residues = [
            r.resid for r, keep in zip(universe.residues, keep_mask) if keep
        ]
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
    orient_mode: str = "normal",
    write_summary: bool = True,
) -> list[Path]:
    """
    Orchestrate Step 1: load system, determine residue orientations (via mwPCA),
    detect GB regions, cluster GB residues and write per-slab GRO + metadata JSON files.

    Parameters
    ----------
    in_gro : str or Path
        Input .gro file.
    resname : str
        Residue name to use for orientation/GB detection (e.g. 'PEN').
    out_dir : str or Path
        Directory where per-slab outputs are written.
    out_base : str
        Base name for outputs (currently unused, kept for future extension).
    thickness_A : float
        Slab thickness in Angstroms (± thickness/2 around GB plane).
    neigh_cutoff : float
        Neighbor cutoff (Å) for smoothing and GB detection.
    smooth_iters : int
        Number of spatial smoothing iterations.
    dbscan_eps : float
        DBSCAN epsilon (Å) for clustering GB COMs.
    dbscan_min_samples : int
        DBSCAN min_samples for GB clustering.
    select_by : str
        'resid' to select slab by COM of residues, otherwise by atom positions.
    unwrap : bool
        If True, apply MDAnalysis unwrap transformation.
    orient_mode : str
        How to define per-residue orientation vector. See `compute_residue_orientations`.
    write_summary : bool
        Whether to write a summary text file.

    Returns
    -------
    List[Path]
        Paths to written slab .gro files.
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

    # --- mwPCA-based orientations with selectable axis mode ---
    coms, normals, resids = compute_residue_orientations(
        u,
        resname,
        orient_mode=orient_mode,
    )
    logger.info(
        "[step1] Molecules used for orientation/GB detection: %d (orient_mode=%s)",
        len(coms),
        orient_mode,
    )

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
            "gb_label": int(lab),
            "cluster_size": int(len(member_idx)),
            "orient_mode": orient_mode,
        }
        meta_path = slab_dir / f"{slab_stem}.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        logs.append(
            f"rank={rank:02d}\tlabel={lab}\tsize={len(member_idx)}\t"
            f"dir={slab_dir.name}\tfile={out_gro.name}\t"
            f"cart={cart_k}({cart_ang:.2f}°)\t"
            f"latt={lat_k}({lat_ang:.2f}°)\t"
            f"orient_mode={orient_mode}"
        )
        written.append(out_gro)

        logger.info("[step1] Wrote slab #%02d → %s", rank, out_gro)

    if write_summary:
        header = "rank\tlabel\tsize\tdir\tfile\tbest_cart\tbest_latt\torient_mode\n"
        summary = header + "\n".join(logs) + "\n"
        summary_path.write_text(summary)
        logger.info(
            "[step1] Done. Slabs written: %d ; summary: %s",
            len(written),
            summary_path,
        )
    else:
        logger.info("[step1] Done. Slabs written: %d ; summary disabled.", len(written))

    return written





#old version depending on PCA normals only
# from __future__ import annotations

# import json
# import logging
# from pathlib import Path
# from typing import Iterable, Tuple

# import numpy as np
# import MDAnalysis as mda
# from MDAnalysis.coordinates.GRO import GROWriter
# from sklearn.cluster import DBSCAN

# from src.utils import (
#     principal_normal,
#     spherical_2means_headless,
#     smooth_labels_spatial,
#     detect_gb_pbc,
#     fit_plane,
#     signed_distance_along_vec,
#     get_cell_vectors_A,
#     axis_report,
# )

# logger = logging.getLogger(__name__)

# __all__ = [
#     "step1_extract_all_slabs",
#     "compute_residue_orientations",
#     "detect_and_cluster_gb_pts",
#     "extract_slab_atoms",
# ]


# def compute_residue_orientations(universe: mda.Universe, resname: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Compute per-residue COMs, principal normals and residue ids for residues matching `resname`.

#     Returns
#     -------
#     coms, normals, resids
#     """
#     pen_res = [r for r in universe.residues if r.resname == resname]
#     if not pen_res:
#         raise RuntimeError(f"No residues named {resname} in provided Universe")

#     heavy_all = universe.select_atoms("not name H*")

#     coms = []
#     normals = []
#     resids = []
#     for r in pen_res:
#         ag = r.atoms
#         pts = (ag & heavy_all).positions
#         if len(pts) < 6:
#             pts = ag.select_atoms("not name H*").positions
#         if len(pts) < 6:
#             # skip very small or malformed residues
#             continue
#         coms.append(ag.center_of_mass())
#         normals.append(principal_normal(pts))
#         resids.append(r.resid)

#     return np.asarray(coms), np.asarray(normals), np.asarray(resids, dtype=int)


# def detect_and_cluster_gb_pts(
#     coms: np.ndarray,
#     normals: np.ndarray,
#     resids: np.ndarray,
#     box: Iterable[float],
#     neigh_cutoff: float,
#     smooth_iters: int,
#     dbscan_eps: float,
#     dbscan_min_samples: int,
# ):
#     """
#     Given per-residue COMs and normals, compute smoothed grain labels, detect GB residues
#     and cluster them using DBSCAN. Returns a sorted list of clusters [(label, member_idx), ...].
#     """
#     labels = spherical_2means_headless(normals, iters=20, seed=0)
#     labels = smooth_labels_spatial(
#         coms, labels, box=box, cutoff=neigh_cutoff, iters=smooth_iters
#     )

#     is_gb = detect_gb_pbc(coms, labels, box=box, cutoff=neigh_cutoff)
#     gb_idx = np.where(is_gb)[0]
#     if gb_idx.size == 0:
#         raise RuntimeError("[step1] No GB contacts found for this system.")

#     gb_pts = coms[gb_idx]
#     gb_labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(
#         gb_pts
#     )

#     clusters = []
#     for lab in sorted(set(gb_labels)):
#         if lab < 0:
#             continue  # noise
#         member_idx = gb_idx[gb_labels == lab]
#         clusters.append((lab, member_idx))

#     clusters.sort(key=lambda x: len(x[1]), reverse=True)
#     return clusters


# def extract_slab_atoms(
#     universe: mda.Universe,
#     ctr: np.ndarray,
#     n: np.ndarray,
#     thickness_A: float,
#     select_by: str = "resid",
# ):
#     """
#     Select and return an MDAnalysis AtomGroup corresponding to the slab (±thickness_A/2 along normal).

#     Parameters
#     ----------
#     universe: MDAnalysis.Universe
#     ctr: center point of the fitted plane
#     n: plane normal
#     thickness_A: slab thickness in Angstroms
#     select_by: 'resid' to select residues by COM, otherwise selects by atom distances
#     """
#     half = 0.5 * thickness_A
#     if select_by.lower() == "resid":
#         res_coms = np.array([r.center_of_mass() for r in universe.residues])
#         d_res = signed_distance_along_vec(res_coms, ctr, n)
#         keep_mask = (d_res >= -half) & (d_res <= half)
#         keep_residues = [r.resid for r, keep in zip(universe.residues, keep_mask) if keep]
#         if keep_residues:
#             sel = "resid " + " ".join(map(str, keep_residues))
#             ag = universe.select_atoms(sel)
#         else:
#             ag = universe.atoms[:0]
#     else:
#         d_atom = signed_distance_along_vec(universe.atoms.positions, ctr, n)
#         keep_atoms = (d_atom >= -half) & (d_atom <= half)
#         ag = universe.atoms[keep_atoms]

#     return ag


# def step1_extract_all_slabs(
#     in_gro: str | Path,
#     resname: str,
#     out_dir: str | Path,
#     out_base: str,
#     thickness_A: float,
#     neigh_cutoff: float,
#     smooth_iters: int,
#     dbscan_eps: float,
#     dbscan_min_samples: int,
#     select_by: str,
#     unwrap: bool,
#     write_summary: bool = True,
# ) -> list[Path]:
#     """
#     Orchestrate Step 1: load system, determine residue orientations, detect GB regions,
#     cluster GB residues and write per-slab GRO + metadata JSON files.

#     Returns
#     -------
#     List[Path] of written slab .gro files.
#     """
#     in_gro = str(in_gro)
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     summary_path = out_dir / "extracted_slabs_summary.txt"
#     logs: list[str] = []

#     logger.info("[step1] Loading full system from: %s", in_gro)
#     u = mda.Universe(in_gro)

#     if unwrap:
#         from MDAnalysis.transformations import unwrap as _unwrap

#         u.trajectory.add_transformations(_unwrap(u.atoms))

#     ts = u.trajectory.ts
#     a_vec, b_vec, c_vec = get_cell_vectors_A(ts)

#     coms, normals, resids = compute_residue_orientations(u, resname)
#     logger.info("[step1] Molecules used for orientation/GB detection: %d", len(coms))

#     clusters = detect_and_cluster_gb_pts(
#         coms=coms,
#         normals=normals,
#         resids=resids,
#         box=u.dimensions,
#         neigh_cutoff=neigh_cutoff,
#         smooth_iters=smooth_iters,
#         dbscan_eps=dbscan_eps,
#         dbscan_min_samples=dbscan_min_samples,
#     )

#     written: list[Path] = []
#     logger.info("[step1] Found %d GB clusters from DBSCAN.", len(clusters))

#     for rank, (lab, member_idx) in enumerate(clusters, start=1):
#         pts = coms[member_idx]
#         ctr, n = fit_plane(pts)

#         cart_k, cart_ang, lat_k, lat_ang = axis_report(n, a_vec, b_vec, c_vec)

#         # select slab atoms either by residue or atom distances
#         if select_by.lower() == "resid":
#             d_res = signed_distance_along_vec(coms, ctr, n)
#             keep_mask = (d_res >= -0.5 * thickness_A) & (d_res <= 0.5 * thickness_A)
#             keep_resids = resids[keep_mask].tolist()
#             if keep_resids:
#                 ag = u.select_atoms("resid " + " ".join(map(str, keep_resids)))
#             else:
#                 ag = u.atoms[:0]
#         else:
#             d_atom = signed_distance_along_vec(u.atoms.positions, ctr, n)
#             keep_atoms = (d_atom >= -0.5 * thickness_A) & (d_atom <= 0.5 * thickness_A)
#             ag = u.atoms[keep_atoms]

#         slab_stem = f"slab_{rank:02d}"
#         slab_dir = out_dir / slab_stem
#         slab_dir.mkdir(parents=True, exist_ok=True)

#         out_gro = slab_dir / f"{slab_stem}.gro"
#         with GROWriter(str(out_gro), n_atoms=len(ag)) as w:
#             w.write(ag)

#         meta = {
#             "ctr": ctr.tolist(),
#             "normal": (n / np.linalg.norm(n)).tolist(),
#             "cart_best_axis": cart_k,
#             "cart_angle_deg": cart_ang,
#             "latt_best_axis": lat_k,
#             "latt_angle_deg": lat_ang,
#         }
#         meta_path = slab_dir / f"{slab_stem}.json"
#         meta_path.write_text(json.dumps(meta, indent=2))

#         logs.append(
#             f"rank={rank:02d}\tlabel={lab}\tsize={len(member_idx)}\t"
#             f"dir={slab_dir.name}\tfile={out_gro.name}\t"
#             f"cart={cart_k}({cart_ang:.2f}°)\t"
#             f"latt={lat_k}({lat_ang:.2f}°)"
#         )
#         written.append(out_gro)

#         logger.info("[step1] Wrote slab #%02d → %s", rank, out_gro)

#     if write_summary:
#         header = "rank\tlabel\tsize\tdir\tfile\tbest_cart\tbest_latt\n"
#         summary = header + "\n".join(logs) + "\n"
#         summary_path.write_text(summary)
#         logger.info("[step1] Done. Slabs written: %d ; summary: %s", len(written), summary_path)
#     else:
#         logger.info("[step1] Done. Slabs written: %d ; summary disabled.", len(written))

#     return written