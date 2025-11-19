"""Module for reading and writing molecular structure files."""


def read_gro(gro_file_path: str) -> list[str]:
    """Read a GRO file and return its lines."""
    if not gro_file_path.endswith(".gro"):
        raise ValueError(" ## gro format is required!")

    with open(gro_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f" # gro file was loaded from {gro_file_path}")

    return lines


def output_gro(residues: list, path: str):
    natoms = sum([len(residue) for residue in residues])
    with open(path, "w", encoding="utf-8") as f:
        f.write("Generated gro file\n")
        f.write(f"{natoms:5d}\n")
        atom_counter = 1
        for residue in residues:
            resid = residue[0][0]
            resname = residue[0][1]
            for atom in residue:
                atomname = atom[2]
                x, y, z = atom[-1]
                f.write(
                    f"{resid:5d}{resname:<5}{atomname:>5}{atom_counter:5d}"
                    f"{x:8.3f}{y:8.3f}{z:8.3f}\n"
                )
                atom_counter += 1
        f.write("   10.00000   10.00000   10.00000\n")  # box vectors


# Other formats should be added later !
