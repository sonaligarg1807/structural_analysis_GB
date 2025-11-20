# misorientation_angle.py
"""
Compute crystallographic misorientation between two grains using the same
lattice vectors (ff/ef) used for contact-plane analysis.

This refactored version preserves the original algorithms and public API,
but is import-safe, uses structured logging and includes concise type hints
and docstrings. No algorithmic logic was changed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.linalg import norm, svd
import logging

# Reuse helpers from contactplanes so we are guaranteed to use the same vectors/logic
from .contactplanes import (
    read_grain_vectors,
    assign_abc,
    compute_com_from_gro,
    normalize as _normalize_vec,
    default_latvec_output_path,
)

logger = logging.getLogger(__name__)

# ========= Optional: orix (recommended) =========
try:
    from orix.quaternion import Orientation, Misorientation  # type: ignore
    from orix.crystal_map import Phase  # type: ignore

    ORIX_AVAILABLE = True
    logger.info("[misori] orix available")
except Exception as e:  # pragma: no cover - import-time may fail when orix missing
    ORIX_AVAILABLE = False
    logger.info("[misori] orix not available (%s). Misorientation will be skipped.", e)


def unit(v: np.ndarray) -> np.ndarray:
    """Normalize a 3D vector; return zero-vector unchanged if norm=0."""
    v = np.asarray(v, dtype=float)
    n = norm(v)
    return v / n if n > 0.0 else v


def orthonormalize(A: np.ndarray) -> np.ndarray:
    """
    Project any 3x3 matrix to the closest proper rotation (SO(3)) via SVD.
    Ensures det=+1. Used to clean numerical drift from rotation-like matrices.
    """
    A = np.asarray(A, dtype=float)
    U, _, Vt = svd(A)
    Rso = U @ Vt
    if np.linalg.det(Rso) < 0.0:
        U[:, -1] *= -1.0
        Rso = U @ Vt
    return Rso


def _build_orix_symmetry(symmetry_name: str):
    """
    Map user-friendly symmetry names to an orix Phase point group.

    Pentacene: triclinic Ci → point group "-1".
    """
    if not ORIX_AVAILABLE:
        return None

    name = symmetry_name.lower().strip()
    if name in {"triclinic", "ci", "-1", "p-1", "p -1"}:
        pg = "-1"  # pentacene
    elif name in {"monoclinic_b", "2/m"}:
        pg = "2/m"
    elif name in {"orthorhombic", "mmm"}:
        pg = "mmm"
    else:
        # conservative default
        pg = "-1"

    phase = Phase(point_group=pg)
    return phase.point_group


def _to_deg_array(x) -> np.ndarray:
    """
    Convert an angle container (orix Angle or raw radians) to float array (degrees).
    """
    if hasattr(x, "degree"):
        arr = np.atleast_1d(x.degree).astype(float)
        return arr
    arr = np.atleast_1d(x).astype(float)
    return np.degrees(arr)


def _axis_to_array(ax) -> np.ndarray:
    """
    Convert an axis container (orix or numpy) to ndarray of shape (N,3).
    """
    if hasattr(ax, "data"):
        arr = np.asarray(ax.data, float)
    else:
        arr = np.asarray(ax, float)
    return np.atleast_2d(arr)


def calculate_misorientation_orix(
    gA: np.ndarray,
    gB: np.ndarray,
    symmetry_name: str = "triclinic",
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute misorientation using orix with the chosen symmetry.

    Returns (angle_deg, Rmat, axis) where:
      - angle_deg is the minimal misorientation angle (degrees)
      - Rmat is the 3x3 misorientation rotation matrix (or None)
      - axis is the misorientation axis (unit vector) in lab frame (or None)

    If orix is not available or the computation fails, returns (None, None, None).
    """
    if not ORIX_AVAILABLE:
        return None, None, None

    # Clean up to proper rotations to avoid numerical garbage
    gA = orthonormalize(gA)
    gB = orthonormalize(gB)

    try:
        sym = _build_orix_symmetry(symmetry_name)
        if sym is None:
            logger.warning("[misori] Could not build symmetry; skipping misorientation.")
            return None, None, None

        OA = Orientation.from_matrix(gA, symmetry=sym)
        OB = Orientation.from_matrix(gB, symmetry=sym)

        # Misorientation M = OB * ~OA
        M = Misorientation(OB * ~OA, symmetry=(sym, sym))

        # Map to symmetry-reduced zone if possible (minimal disorientation)
        try:
            Mred = M.map_into_symmetry_reduced_zone()
        except Exception:
            Mred = M

        ang_deg = _to_deg_array(Mred.angle)
        axes = _axis_to_array(Mred.axis)

        idx = int(np.argmin(ang_deg))
        angle_deg = float(ang_deg[idx])
        axis = unit(axes[idx])

        # Get misorientation matrix (may be single matrix or array)
        Rmat = None
        try:
            Rm = Mred.as_matrix()
            Rm = np.asarray(Rm)
            if Rm.ndim == 3:
                Rmat = Rm[idx]
            else:
                Rmat = Rm
        except Exception:
            Rmat = None

        return angle_deg, Rmat, axis

    except Exception as e:
        logger.exception("[misori] orix misorientation failed: %s", e)
        return None, None, None


def misorientation_from_frames(
    gA: np.ndarray,
    gB: np.ndarray,
    gb_normal: Optional[np.ndarray] = None,
    symmetry_name: str = "triclinic",
) -> Dict[str, Optional[float | np.ndarray]]:
    """
    Compute misorientation between two grain frames using orix, and optionally
    decompose into twist/tilt relative to the GB normal.

    Parameters
    ----------
    gA, gB : 3x3 arrays
        Grain frames (columns = a,b,c) in lab coordinates.
    gb_normal : optional 3-vector
        GB plane normal in lab coords. If provided, compute twist/tilt decomposition.
    symmetry_name : str
        Symmetry label for orix.

    Returns
    -------
    dict with keys:
      - "theta_deg", "axis", "twist_deg", "tilt_deg", "method"
    """
    angle_deg, R_delta, axis = calculate_misorientation_orix(
        gA, gB, symmetry_name=symmetry_name
    )

    if angle_deg is None or axis is None:
        return {
            "theta_deg": None,
            "axis": None,
            "twist_deg": None,
            "tilt_deg": None,
            "method": "none",
        }

    if gb_normal is not None:
        gbn = unit(gb_normal)
        twist_deg = abs(float(np.dot(axis, gbn))) * angle_deg
        tilt_sq = max(angle_deg**2 - twist_deg**2, 0.0)
        tilt_deg = float(np.sqrt(tilt_sq))
    else:
        twist_deg = None
        tilt_deg = None

    return {
        "theta_deg": float(angle_deg),
        "axis": axis,
        "twist_deg": twist_deg,
        "tilt_deg": tilt_deg,
        "method": "orix",
    }


def misorientation_for_group(
    g1_gro_file,
    g2_gro_file,
    g1_txt: Optional[str | Path] = None,
    g2_txt: Optional[str | Path] = None,
    symmetry_name: str = "triclinic",
) -> Dict[str, Optional[float | np.ndarray]]:
    """
    High-level convenience wrapper for the GB pipeline.

    Mirrors contactplanes_for_group inputs and returns the same dict format as
    misorientation_from_frames().
    """
    g1_gro_file = Path(g1_gro_file)
    g2_gro_file = Path(g2_gro_file)

    # Default latvec txt paths (consistent with contactplanes_for_group)
    if g1_txt is None:
        g1_txt = default_latvec_output_path(g1_gro_file)
    if g2_txt is None:
        g2_txt = default_latvec_output_path(g2_gro_file)

    g1_txt = Path(g1_txt)
    g2_txt = Path(g2_txt)

    # Compute COMs and GB normal as in contactplanes.py
    com1 = compute_com_from_gro(g1_gro_file)
    com2 = compute_com_from_gro(g2_gro_file)
    conn_vec = com2 - com1
    if norm(conn_vec) < 1e-10:
        logger.error("[misori] Grains overlap, cannot compute GB normal.")
        return {
            "theta_deg": None,
            "axis": None,
            "twist_deg": None,
            "tilt_deg": None,
            "method": "none",
        }
    gb_normal = _normalize_vec(conn_vec)

    # Read ff and ef vectors (same as contact-plane logic)
    ff1, ef1 = read_grain_vectors(str(g1_txt))
    ff2, ef2 = read_grain_vectors(str(g2_txt))

    if ff1 is None or ef1 is None or ff2 is None or ef2 is None:
        logger.warning("[misori] Missing ff/ef vectors; skipping misorientation.")
        return {
            "theta_deg": None,
            "axis": None,
            "twist_deg": None,
            "tilt_deg": None,
            "method": "none",
        }

    # Assign a,b,c for each grain and build frames: columns = (a, b, c)
    a1, b1, c1 = assign_abc(ff1, ef1)
    a2, b2, c2 = assign_abc(ff2, ef2)
    g1_frame = np.column_stack([a1, b1, c1])
    g2_frame = np.column_stack([a2, b2, c2])

    # Compute misorientation using orix and decompose into twist/tilt
    miso = misorientation_from_frames(
        g1_frame,
        g2_frame,
        gb_normal=gb_normal,
        symmetry_name=symmetry_name,
    )

    if miso["theta_deg"] is not None:
        logger.info(
            "[misori] Θ = %.2f°, twist = %s, tilt = %s",
            miso["theta_deg"],
            miso["twist_deg"],
            miso["tilt_deg"],
        )
    else:
        logger.info("[misori] Misorientation not computed (orix missing or input issue).")

    return miso