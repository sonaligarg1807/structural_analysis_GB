# contactplanes.py
import numpy as np
import MDAnalysis as mda
from pathlib import Path


def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-12 else v


def read_grain_vectors(filename):
    """
    Reads ff and ef vectors from file: idx type distance vec_x vec_y vec_z
    Returns (ff_vector, ef_vector), each as np.ndarray or None.
    """
    ff_vec, ef_vec = None, None
    with open(filename, "r") as f:
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


def assign_abc(ff_vec, ef_vec):
    """
    Assign a, b, c axes from ff and ef vectors.
    a: ff direction
    b: ef direction
    c: a × b
    """
    a = normalize(ff_vec)
    b = normalize(ef_vec)
    c = normalize(np.cross(a, b))
    return a, b, c


def compute_com_from_gro(gro_file):
    """
    Compute center of mass of all atoms in a .gro file using MDAnalysis
    """
    u = mda.Universe(str(gro_file))
    return u.atoms.center_of_mass()


def contactplan(a, b, c, contact_vec):
    """
    Given lattice axes a,b,c and a contact vector, determine which plane
    (ab, ac, bc) the interface normal is closest to.
    """
    # Compute plane normals
    n_ab = normalize(np.cross(a, b))
    n_ac = normalize(np.cross(a, c))
    n_bc = normalize(np.cross(b, c))

    # Dot products
    dot_ab = abs(np.dot(contact_vec, n_ab))
    dot_ac = abs(np.dot(contact_vec, n_ac))
    dot_bc = abs(np.dot(contact_vec, n_bc))

    print(f" # Dot product with n_ab: {dot_ab:.4f}")
    print(f" # Dot product with n_ac: {dot_ac:.4f}")
    print(f" # Dot product with n_bc: {dot_bc:.4f}")

    dots = [dot_ab, dot_ac, dot_bc]
    planes = ["ab", "ac", "bc"]
    return planes[np.argmax(dots)]


def two_grains_contact_from_gro(g1_gro_file, g2_gro_file, g1_txt, g2_txt):
    """
    Original main function: given two grain .gro files and their ff/ef
    txt outputs, compute contact planes for both grains.

    Returns
    -------
    (g1_plane, g2_plane)
    """
    # Compute COMs from .gro files
    com1 = compute_com_from_gro(g1_gro_file)
    com2 = compute_com_from_gro(g2_gro_file)
    print(f" # COM of Grain 1: {com1}")
    print(f" # COM of Grain 2: {com2}")

    # Contact vector
    conn_vec = com2 - com1
    if np.linalg.norm(conn_vec) < 1e-10:
        print(" # ERROR: Grains overlap, cannot compute contact plane.")
        return None, None
    contact_vec = normalize(conn_vec)
    print(f" # Contact vector: {contact_vec}")

    # Read ff and ef vectors
    ff1, ef1 = read_grain_vectors(g1_txt)
    ff2, ef2 = read_grain_vectors(g2_txt)

    # Assign a,b,c for each grain
    a1, b1, c1 = assign_abc(ff1, ef1)
    a2, b2, c2 = assign_abc(ff2, ef2)

    # Compute contact planes
    g1_plane = contactplan(a1, b1, c1, contact_vec)
    g2_plane = contactplan(a2, b2, c2, contact_vec)

    print(f" # Grain 1 contact plane: {g1_plane}")
    print(f" # Grain 2 contact plane: {g2_plane}")
    print(" # Finished contact plane analysis.")

    return g1_plane, g2_plane


def default_latvec_output_path(gro_path):
    """
    Given a grain .gro path, return the default latvec output txt path:
      <stem>_output.txt
    This matches analyze_grain_latvecs() behaviour.
    """
    p = Path(gro_path)
    return p.with_name(f"{p.stem}_output.txt")


def contactplanes_for_group(g1_gro_file, g2_gro_file, g1_txt=None, g2_txt=None):
    """
    Convenience wrapper for pipeline:
      - If g1_txt / g2_txt not given, use <stem>_output.txt next to the .gro
      - Call two_grains_contact_from_gro
    """
    g1_gro_file = Path(g1_gro_file)
    g2_gro_file = Path(g2_gro_file)

    if g1_txt is None:
        g1_txt = default_latvec_output_path(g1_gro_file)
    if g2_txt is None:
        g2_txt = default_latvec_output_path(g2_gro_file)

    return two_grains_contact_from_gro(str(g1_gro_file), str(g2_gro_file), str(g1_txt), str(g2_txt))
