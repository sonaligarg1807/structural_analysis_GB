"""Module for parsing GRO files and extracting residue information."""

from dataclasses import dataclass
import numpy as np


def gro_resid(line: str) -> int:
    """Extract residue ID from a GRO file line."""
    return int(line[0:5].strip())


def gro_resname(line: str) -> str:
    """Extract residue name from a GRO file line."""
    return line[5:10].strip()


def gro_atomname(line: str) -> str:
    """Extract atom name from a GRO file line."""
    return line[10:15].strip()


def gro_atomnr(line: str) -> int:
    """Extract atom number from a GRO file line."""
    return int(line[15:20].strip())


def gro_atom_pos(line: str) -> np.ndarray:
    """Extract atom position from a GRO file line."""
    x = float(line[20:28].strip())
    y = float(line[28:36].strip())
    z = float(line[36:44].strip())
    return np.array([x, y, z])


@dataclass
class Atom:
    """Data class to hold atom information from a GRO file line."""

    resid: int
    resname: str
    atomname: str
    atomnr: int
    position: np.ndarray


@dataclass
class Residue:
    """Data class to hold residue information."""

    resid: int
    resname: str
    atoms: list[Atom]


def make_atom(gro_line: str) -> Atom:
    """Parse a line from a GRO file and return its components."""
    atom = Atom(
        resid=gro_resid(gro_line),
        resname=gro_resname(gro_line),
        atomname=gro_atomname(gro_line),
        atomnr=gro_atomnr(gro_line),
        position=gro_atom_pos(gro_line),
    )
    return atom


def make_residue(atoms: list[Atom], resid: int) -> Residue:
    """Create a Residue object from a list of Atom objects with the same resid."""
    res_atoms = [atom for atom in atoms if atom.resid == resid]
    if not res_atoms:
        raise ValueError(f"No atoms found for residue ID {resid}")
    resname = res_atoms[0].resname
    residue = Residue(resid=resid, resname=resname, atoms=res_atoms)
    return residue


def split_gro2residues(atom_lines: list[str]) -> list[Residue]:
    """Create a Residue object from a list of Atom objects."""
    atoms = [make_atom(line) for line in atom_lines]
    resids = set(atom.resid for atom in atoms)
    residues = [make_residue(atoms, resid) for resid in resids]
    return residues


# def split_gro2resides(
#     atom_lines: list[str],
# ) -> list[Residue]:
#     """Split GRO atom lines into residues."""
#     residues = []
#     for res_id, group in it.groupby(atom_lines, key=gro_resid):
#         atom = [parse_gro_line(line) for line in group]
#         residues.append(atom)
#     return residues
