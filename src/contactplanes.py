# contactplanes.py
"""
Contact-plane inference between two grains.

This module is a reorganized copy of the original code:
- identical computations and outputs
- added light documentation and logging
- kept default filenames and wrapper behavior unchanged
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import MDAnalysis as mda
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["contactplanes_for_group", "two_grains_contact_from_gro", "default_latvec_output_path"]

EPS = 1e-12


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > EPS else v


def read_grain_vectors(filename: Path | str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Reads ff and ef vectors from file with format:
      idx type distance vec_x vec_y vec_z

    Returns (ff_vector, ef_vector) where each is np.ndarray or None if missing.
    """
    ff_vec: Optional[np.ndarray] = None
    ef_vec: Optional[np.ndarray] = None
    filename = Path(filename)
    with filename.open("r") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            _, gtype, _, x, y, z = parts
            vec = np.array([float(x), float(y), float(z)])
            if gtype == "ff":
                ff_vec = vec
            elif gtype == "ef":
                ef_vec = vec
    return ff_vec, ef_vec


def assign_abc(ff_vec: np.ndarray, ef_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign a, b, c axes from ff and ef vectors:
      - a: normalized ff direction
      - b: normalized ef direction
      - c: normalized cross(a, b)
    """
    a = normalize(ff_vec)
    b = normalize(ef_vec)
    c = normalize(np.cross(a, b))
    return a, b, c


def compute_com_from_gro(gro_file: Path | str) -> np.ndarray:
    """
    Compute center of mass of atoms in a .gro file using MDAnalysis.
    """
    u = mda.Universe(str(gro_file))
    return u.atoms.center_of_mass()


def contactplan(a: np.ndarray, b: np.ndarray, c: np.ndarray, contact_vec: np.ndarray) -> str:
    """
    Given lattice axes a,b,c and a contact vector, determine which plane
    (ab, ac, bc) the interface normal is closest to.
    """
    n_ab = normalize(np.cross(a, b))
    n_ac = normalize(np.cross(a, c))
    n_bc = normalize(np.cross(b, c))

    dot_ab = abs(np.dot(contact_vec, n_ab))
    dot_ac = abs(np.dot(contact_vec, n_ac))
    dot_bc = abs(np.dot(contact_vec, n_bc))

    logger.debug("Dot product with n_ab: %.4f", dot_ab)
    logger.debug("Dot product with n_ac: %.4f", dot_ac)
    logger.debug("Dot product with n_bc: %.4f", dot_bc)

    dots = [dot_ab, dot_ac, dot_bc]
    planes = ["ab", "ac", "bc"]
    return planes[int(np.argmax(dots))]


def two_grains_contact_from_gro(
    g1_gro_file: Path | str, g2_gro_file: Path | str, g1_txt: Path | str, g2_txt: Path | str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Compute contact plane labels for two grains given grain .gro files and their latvec outputs.

    Returns (g1_plane, g2_plane) where each element is one of "ab","ac","bc" or None on failure.
    """
    com1 = compute_com_from_gro(g1_gro_file)
    com2 = compute_com_from_gro(g2_gro_file)
    logger.debug("COM of Grain 1: %s", com1)
    logger.debug("COM of Grain 2: %s", com2)

    conn_vec = com2 - com1
    if np.linalg.norm(conn_vec) < 1e-10:
        logger.error("Grains overlap, cannot compute contact plane.")
        return None, None
    contact_vec = normalize(conn_vec)
    logger.debug("Contact vector: %s", contact_vec)

    ff1, ef1 = read_grain_vectors(Path(g1_txt))
    ff2, ef2 = read_grain_vectors(Path(g2_txt))

    if ff1 is None or ef1 is None or ff2 is None or ef2 is None:
        logger.warning("Missing ff/ef vectors for one of the grains; results may be unreliable.")

    a1, b1, c1 = assign_abc(ff1, ef1)  # type: ignore[arg-type]
    a2, b2, c2 = assign_abc(ff2, ef2)  # type: ignore[arg-type]

    g1_plane = contactplan(a1, b1, c1, contact_vec)
    g2_plane = contactplan(a2, b2, c2, contact_vec)

    logger.info("Grain 1 contact plane: %s", g1_plane)
    logger.info("Grain 2 contact plane: %s", g2_plane)
    return g1_plane, g2_plane


def default_latvec_output_path(gro_path: Path | str) -> Path:
    """
    Default latvec output filename for a grain .gro:
      <stem>_latvecs.txt
    """
    p = Path(gro_path)
    return p.with_name(f"{p.stem}_latvecs.txt")


def contactplanes_for_group(g1_gro_file: Path | str, g2_gro_file: Path | str, g1_txt: Optional[Path | str] = None, g2_txt: Optional[Path | str] = None):
    """
    Convenience wrapper: determine latvec txt paths if not provided and compute contact planes.
    """
    g1_gro_file = Path(g1_gro_file)
    g2_gro_file = Path(g2_gro_file)

    if g1_txt is None:
        g1_txt = default_latvec_output_path(g1_gro_file)
    if g2_txt is None:
        g2_txt = default_latvec_output_path(g2_gro_file)

    return two_grains_contact_from_gro(str(g1_gro_file), str(g2_gro_file), str(g1_txt), str(g2_txt))