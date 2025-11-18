# utils.py
import re
import warnings
from collections import deque

import numpy as np
from numpy.linalg import eigh, svd, norm
from sklearn.neighbors import NearestNeighbors
from MDAnalysis.lib.distances import capped_distance


def unit(v):
    v = np.asarray(v, float)
    n = norm(v)
    return v / n if n > 1e-12 else v


def principal_normal(points_xyz):
    P = points_xyz - points_xyz.mean(0)
    C = P.T @ P
    evals, evecs = eigh(C)
    n = unit(evecs[:, np.argmin(evals)])
    if n[2] < 0:
        n = -n
    return n


def nematic_director(normals):
    S = normals.T @ normals / len(normals)
    evals, evecs = eigh(S)
    d = evecs[:, np.argmax(evals)]
    if d[2] < 0:
        d = -d
    return unit(d)


def spherical_2means_headless(normals, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=len(normals))
    for _ in range(iters):
        d0 = nematic_director(normals[labels == 0]) if np.any(labels == 0) else unit(rng.normal(size=3))
        d1 = nematic_director(normals[labels == 1]) if np.any(labels == 1) else unit(rng.normal(size=3))
        s0 = np.abs(normals @ d0)
        s1 = np.abs(normals @ d1)
        labels = (s1 > s0).astype(int)
    # final centers (unused but kept for completeness)
    _ = nematic_director(normals[labels == 0]) if np.any(labels == 0) else unit(rng.normal(size=3))
    _ = nematic_director(normals[labels == 1]) if np.any(labels == 1) else unit(rng.normal(size=3))
    return labels


def smooth_labels_spatial(coms, labels, box, cutoff=5.0, iters=2):
    labels = labels.copy()
    for _ in range(iters):
        pairs = capped_distance(coms, coms, max_cutoff=cutoff, box=box, return_distances=False)
        neigh = [[] for _ in range(len(coms))]
        for i, j in pairs:
            neigh[i].append(j)
            neigh[j].append(i)
        new = labels.copy()
        for i, lst in enumerate(neigh):
            if not lst:
                continue
            votes = np.bincount([labels[i]] + [labels[j] for j in lst], minlength=2)
            maj = np.argmax(votes)
            if votes[maj] >= 0.6 * (len(lst) + 1):
                new[i] = maj
        labels = new
    return labels


def detect_gb_pbc(coms, labels, box, cutoff=5.0):
    N = len(coms)
    is_gb = np.zeros(N, dtype=bool)
    pairs = capped_distance(coms, coms, max_cutoff=cutoff, box=box, return_distances=False)
    for i, j in pairs:
        if labels[i] != labels[j]:
            is_gb[i] = True
            is_gb[j] = True
    return is_gb


def fit_plane(points):
    ctr = points.mean(0)
    P = points - ctr
    _, _, vh = svd(P, full_matrices=False)
    n = unit(vh[-1])
    return ctr, n


def signed_distance_along_vec(x, ctr, vhat):
    x = np.asarray(x, float)
    return (x - ctr) @ vhat


def get_cell_vectors_A(ts):
    tri = getattr(ts, "triclinic_dimensions", None)
    if tri is not None and np.shape(tri) == (3, 3):
        return tri[0], tri[1], tri[2]
    Lx, Ly, Lz, alpha, beta, gamma = ts.dimensions
    ar, br, gr = np.deg2rad([alpha, beta, gamma])
    a = np.array([Lx, 0.0, 0.0])
    b = np.array([Ly * np.cos(gr), Ly * np.sin(gr), 0.0])
    cx = Lz * np.cos(br)
    cy = Lz * (np.cos(ar) - np.cos(br) * np.cos(gr)) / max(np.sin(gr), 1e-12)
    cz = np.sqrt(max(Lz**2 - cx**2 - cy**2, 0.0))
    c = np.array([cx, cy, cz])
    return a, b, c


def axis_report(n, a_vec, b_vec, c_vec):
    cart = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
    latt = {"a": a_vec, "b": b_vec, "c": c_vec}

    def best(dct):
        best_k, best_c = None, -1
        for k, v in dct.items():
            c = abs(np.dot(unit(n), unit(v)))
            if c > best_c:
                best_k, best_c = k, c
        ang = float(np.degrees(np.arccos(np.clip(best_c, 0, 1))))
        return best_k, ang

    cart_k, cart_ang = best(cart)
    lat_k, lat_ang = best(latt)
    return cart_k, cart_ang, lat_k, lat_ang


def pbc_radius_neighbors(positions, radius, box, include_self=False):
    pairs = capped_distance(positions, positions, max_cutoff=radius, box=box)[0]
    N = positions.shape[0]
    neigh = [[] for _ in range(N)]
    for i, j in pairs:
        if include_self or i != j:
            neigh[i].append(j)
        if include_self or j != i:
            neigh[j].append(i)
    return [np.array(n, dtype=int) for n in neigh]


def gyration_long_axis(ag):
    pos = ag.positions - ag.center_of_mass()
    if pos.shape[0] < 3:
        return unit(pos[-1] - pos[0]) if pos.shape[0] >= 2 else np.array([1.0, 0.0, 0.0])
    G = (pos.T @ pos) / pos.shape[0]
    evals, evecs = np.linalg.eigh(G)
    return unit(evecs[:, np.argmax(evals)])


def nematic_embed(u):
    U = np.outer(u, u)
    return np.array([U[0, 0], U[1, 1], U[2, 2], U[0, 1], U[0, 2], U[1, 2]], float)


def nematic_center_to_director(center6d):
    xx, yy, zz, xy, xz, yz = center6d
    M = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], float)
    M = 0.5 * (M + M.T)
    evals, evecs = np.linalg.eigh(M)
    return unit(evecs[:, np.argmax(evals)])


def write_gro_for_resids(u, resid_list, out_path, resid_to_idx):
    from MDAnalysis.coordinates.GRO import GROWriter

    if not resid_list:
        return False
    idx = [resid_to_idx.get(r, None) for r in resid_list]
    idx = [i for i in idx if i is not None]
    if not idx:
        return False
    ag = u.residues[idx].atoms
    if ag.n_atoms == 0:
        return False
    ag.write(out_path)
    return True


def axis_vals_dict(COM):
    return {"x": COM[:, 0], "y": COM[:, 1], "z": COM[:, 2]}


def unitcell_delta_for_axis(axis, A_LEN, B_LEN, C_LEN):
    if axis == "x":
        return float(A_LEN)
    if axis == "y":
        return float(B_LEN)
    if axis == "z":
        return float(C_LEN)
    raise ValueError(f"Axis must be x/y/z, got {axis!r}")


def midlayer_center(arr):
    amin, amax = np.min(arr), np.max(arr)
    return 0.5 * (amin + amax)


def margin_mask_1d(arr, margin):
    amin, amax = np.min(arr), np.max(arr)
    lo, hi = amin + margin, amax - margin
    return (arr >= lo) & (arr <= hi)


def wrap_axis_to_center(vals, box_len, center):
    if not np.isfinite(box_len) or box_len <= 0:
        return vals
    return center + ((vals - center + 0.5 * box_len) % box_len) - 0.5 * box_len


def layer_mask(vals_unwrapped, layer_center, thickness, box_len_axis):
    half = 0.5 * thickness
    vals_wrapped = wrap_axis_to_center(vals_unwrapped, box_len_axis, layer_center)
    return np.abs(vals_wrapped - layer_center) <= half


def connected_components(COM, radius, subset_idx=None):
    """
    Connected components over points with radius-neighbor graph.
    Returns a list of np.array indices.
    """
    if subset_idx is None:
        subset_idx = np.arange(len(COM), dtype=int)
    subset_idx = np.asarray(subset_idx, dtype=int)
    if subset_idx.size == 0:
        return []

    nbrs = NearestNeighbors(radius=radius).fit(COM[subset_idx])
    G = nbrs.radius_neighbors_graph(COM[subset_idx], mode="connectivity")
    seen = np.zeros(subset_idx.size, dtype=bool)
    comps = []
    for s in range(subset_idx.size):
        if seen[s]:
            continue
        q = deque([s])
        seen[s] = True
        cur = [s]
        while q:
            i = q.popleft()
            for j in G[i].indices:
                if not seen[j]:
                    seen[j] = True
                    q.append(j)
                    cur.append(j)
        comps.append(subset_idx[np.array(cur, int)])
    return comps
