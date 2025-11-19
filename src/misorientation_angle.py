# misorientation_angle.py
"""
Compute crystallographic misorientation between two grains using the same
lattice vectors (ff/ef) used for contact-plane analysis.

This module is OPTIONAL and is meant to be run *after* you have run
lattice-vector analysis (latvecs) and contact-plane analysis for a GB group.

Main usages
-----------
1) Low-level (frames already known):

   miso = misorientation_from_frames(gA, gB, gb_normal, symmetry_name="triclinic")

   where gA, gB are 3x3 rotation matrices (columns = a,b,c in lab frame).

2) High-level (pipeline-friendly, same inputs as contactplanes_for_group):

   miso = misorientation_for_group(
       g1_gro_file, g2_gro_file,
       g1_txt=None, g2_txt=None,
       symmetry_name="triclinic"
   )

   - If g1_txt/g2_txt are None, uses "<stem>_output.txt" next to each .gro
     (same convention as analyze_grain_latvecs and contactplanes_for_group).
   - Uses COM(g1), COM(g2) to define GB normal (com2 - com1).
   - Reads ff/ef from those txt files, builds grain frames via assign_abc.

Return value
------------
Both functions return a dict with keys:

    - "theta_deg" : total misorientation angle (degrees) or None
    - "axis"      : 3-vector (unit) or None
    - "twist_deg" : twist component about GB normal (deg) or None
    - "tilt_deg"  : tilt component (deg) or None
    - "method"    : "orix" if computed, otherwise "none"

If orix is not installed or something fails, all values are None and method="none".
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.linalg import norm, svd
from pathlib import Path

# Reuse helpers from contactplanes so we are guaranteed to use the same vectors/logic
from .contactplanes import (
    read_grain_vectors,
    assign_abc,
    compute_com_from_gro,
    normalize as _normalize_vec,
    default_latvec_output_path,
)

# ========= Optional: orix (recommended) =========
try:
    from orix.quaternion import Orientation, Misorientation
    from orix.crystal_map import Phase

    ORIX_AVAILABLE = True
    print("[misori] orix available")
except Exception as e:  # pragma: no cover - import-time
    ORIX_AVAILABLE = False
    print(f"[misori] orix not available ({e}). Misorientation will be skipped.")


# ---------------------------------------------------------------------------
# Small linear-algebra helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# orix symmetry + misorientation
# ---------------------------------------------------------------------------

def _build_orix_symmetry(symmetry_name: str):
    """
    Map user-friendly symmetry names to an orix Phase point group.

    Pentacene: triclinic Ci → point group "-1".
    """
    if not ORIX_AVAILABLE:
        return None

    name = symmetry_name.lower().strip()
    if name in {"triclinic", "ci", "-1", "p-1", "p -1"}:
        pg = "-1"     # pentacene
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

    Parameters
    ----------
    gA, gB
        3x3 rotation matrices (grain frames) in the lab frame.
        Columns should be an orthonormal basis for each grain.
        Here we typically use columns = (a, b, c) from ff/ef analysis.
    symmetry_name
        A string describing the crystal symmetry, e.g. "triclinic", "2/m", "mmm".

    Returns
    -------
    angle_deg
        Minimal misorientation angle in degrees (or None if orix unavailable).
    Rmat
        3x3 rotation matrix for the misorientation (or None).
    axis
        3-vector (unit) for the misorientation axis in the lab frame (or None).
    """
    if not ORIX_AVAILABLE:
        return None, None, None

    # Clean up to proper rotations to avoid numerical garbage
    gA = orthonormalize(gA)
    gB = orthonormalize(gB)

    try:
        sym = _build_orix_symmetry(symmetry_name)
        if sym is None:
            print("[misori] Could not build symmetry; skipping misorientation.")
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

        # Get misorientation matrix
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
        print(f"[misori] orix misorientation failed: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# High-level helper: from frames + GB normal
# ---------------------------------------------------------------------------

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
    gA, gB
        3x3 grain frames (rotation matrices) in lab coordinates.
        These should be the same grain frames you used for contact-plane
        analysis. For this package, we typically build them as:

            g = [a b c]

        where a,b,c are the lattice vectors (ff,ef,a×b).
    gb_normal
        3-vector for the GB plane normal in lab coordinates.
        If provided, we compute:
            - twist_deg = |axis · gb_normal| * Θ
            - tilt_deg  = sqrt(Θ^2 - twist_deg^2)
        If None, twist_deg and tilt_deg are returned as None.
    symmetry_name
        Crystal symmetry string for orix (e.g. "triclinic", "2/m", "mmm").

    Returns
    -------
    dict with keys:
        - "theta_deg" : total misorientation angle Θ (degrees) or None
        - "axis"      : 3-vector (unit) or None
        - "twist_deg" : twist component about GB normal (deg) or None
        - "tilt_deg"  : tilt component (deg) or None
        - "method"    : "orix" if computed, otherwise "none"
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

    twist_deg: Optional[float]
    tilt_deg: Optional[float]

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


# ---------------------------------------------------------------------------
# Pipeline-level helper: from the same grain files as contactplanes_for_group
# ---------------------------------------------------------------------------

def misorientation_for_group(
    g1_gro_file,
    g2_gro_file,
    g1_txt: Optional[str | Path] = None,
    g2_txt: Optional[str | Path] = None,
    symmetry_name: str = "triclinic",
) -> Dict[str, Optional[float | np.ndarray]]:
    """
    High-level convenience wrapper for the GB pipeline.

    It mirrors the I/O style of contactplanes_for_group:

      - If g1_txt / g2_txt are not given, uses "<stem>_output.txt" next
        to the .gro files (same as analyze_grain_latvecs()).
      - Uses COM(g1) and COM(g2) from the .gro files (via MDAnalysis) to
        define the GB normal as com2 - com1 (same as contactplanes.py).
      - Reads ff/ef vectors from the respective *_output.txt files via
        `read_grain_vectors`.
      - Builds grain frames g1, g2 as [a b c] using `assign_abc`.
      - Calls `misorientation_from_frames` and returns its dict.

    Returns
    -------
    miso : dict
        Same as misorientation_from_frames().

    Notes
    -----
    This is intended to be **optional**:
      - You only call it for GB groups where you have already run
        latvecs + contact-plane analysis.
      - If orix is not available, all values will be None and method="none".
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
        print("[misori] ERROR: Grains overlap, cannot compute GB normal.")
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
        print("[misori] Missing ff/ef vectors; skipping misorientation.")
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

    # Small log for debug
    if miso["theta_deg"] is not None:
        print(
            f"[misori] Θ = {miso['theta_deg']:.2f}°, "
            f"twist = {miso['twist_deg']}, tilt = {miso['tilt_deg']}"
        )
    else:
        print("[misori] Misorientation not computed (orix missing or input issue).")

    return miso
