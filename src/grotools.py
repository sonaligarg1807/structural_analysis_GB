"""Utilities for parsing GRO files and representing atoms/residues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


# --- low-level field extractors -------------------------------------------------


def gro_resid(line: str) -> int:
    """Extract residue ID from a GRO atom line."""
    return int(line[0:5].strip())


def gro_resname(line: str) -> str:
    """Extract residue name from a GRO atom line."""
    return line[5:10].strip()


def gro_atomname(line: str) -> str:
    """Extract atom name from a GRO atom line."""
    return line[10:15].strip()


def gro_atomnr(line: str) -> int:
    """Extract atom number from a GRO atom line."""
    return int(line[15:20].strip())


def gro_atom_pos(line: str) -> np.ndarray:
    """Extract atom position (x, y, z) from a GRO atom line."""
    x = float(line[20:28].strip())
    y = float(line[28:36].strip())
    z = float(line[36:44].strip())
    return np.array([x, y, z], dtype=np.float32)


# --- data structures ------------------------------------------------------------


@dataclass(slots=True)
class Atom:
    """Atom information parsed from a GRO atom line."""

    resid: int
    resname: str
    atomname: str
    atomnr: int
    position: np.ndarray  # shape (3,)


@dataclass(slots=True)
class Residue:
    """Residue = group of atoms sharing the same resid."""

    resid: int
    resname: str
    atoms: List[Atom]


# --- constructors ---------------------------------------------------------------


def parse_atom_line(gro_line: str) -> Atom:
    """Parse a single GRO atom line into an Atom object."""
    return Atom(
        resid=gro_resid(gro_line),
        resname=gro_resname(gro_line),
        atomname=gro_atomname(gro_line),
        atomnr=gro_atomnr(gro_line),
        position=gro_atom_pos(gro_line),
    )


def make_residue(atoms: List[Atom], resid: int) -> Residue:
    """
    Create a Residue object from a list of Atom objects with the same resid.

    NOTE: For most workflows you'll just call `split_gro2residues` instead.
    """
    res_atoms = [atom for atom in atoms if atom.resid == resid]
    if not res_atoms:
        raise ValueError(f"No atoms found for residue ID {resid}")
    resname = res_atoms[0].resname
    return Residue(resid=resid, resname=resname, atoms=res_atoms)


def split_gro2residues(atom_lines: Iterable[str]) -> List[Residue]:
    """
    Split GRO atom lines into Residue objects.

    Assumes that atoms are ordered by residue ID (standard for GROMACS .gro
    where each residue is contiguous). This makes the grouping O(N) instead of
    O(N^2).

    Parameters
    ----------
    atom_lines
        Lines containing only atom records, i.e. `gro[2:-1]`.

    Returns
    -------
    list[Residue]
    """
    residues: List[Residue] = []
    current_resid: int | None = None
    current_atoms: List[Atom] = []

    for line in atom_lines:
        line = line.rstrip("\n")
        if not line:
            continue

        atom = parse_atom_line(line)

        if current_resid is None:
            # first atom
            current_resid = atom.resid
            current_atoms = [atom]
            continue

        if atom.resid != current_resid:
            # flush previous residue
            residues.append(
                Residue(
                    resid=current_resid,
                    resname=current_atoms[0].resname,
                    atoms=current_atoms,
                )
            )
            # start new residue
            current_resid = atom.resid
            current_atoms = [atom]
        else:
            current_atoms.append(atom)

    # flush last residue
    if current_resid is not None and current_atoms:
        residues.append(
            Residue(
                resid=current_resid,
                resname=current_atoms[0].resname,
                atoms=current_atoms,
            )
        )

    return residues
