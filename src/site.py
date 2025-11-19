"""Module defining the Site class for molecular residues."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as Rot

from .grotools import Residue


mass_ref = {"H": 1.001, "C": 12.001}  # extend if needed


class Site:
    """
    Class representing a molecular site (one residue) with geometric properties.

    Provides:
      - atom labels
      - coordinates
      - masses
      - center of mass
      - mass-weighted PCA axes (long / short / normal)
    """

    def __init__(self, residue: Residue) -> None:
        self.residue = residue

        # lazy-cache attributes
        self._atomlbls: List[str] | None = None
        self._crds: np.ndarray | None = None
        self._masses: np.ndarray | None = None
        self._com: np.ndarray | None = None

    # ------------------------------------------------------------------
    # basic identity
    # ------------------------------------------------------------------
    @property
    def resid(self) -> int:
        """Residue ID."""
        return self.residue.resid

    # ------------------------------------------------------------------
    # basic per-atom data
    # ------------------------------------------------------------------
    @property
    def atomlbls(self) -> List[str]:
        """List of element labels derived from atom names (first character)."""
        if self._atomlbls is None:
            self._atomlbls = [atom.atomname[0] for atom in self.residue.atoms]
        return self._atomlbls

    @property
    def crds(self) -> np.ndarray:
        """(N, 3) array of atomic coordinates."""
        if self._crds is None:
            self._crds = np.array(
                [atom.position for atom in self.residue.atoms],
                dtype=np.float32,
            )
        return self._crds

    @property
    def masses(self) -> np.ndarray:
        """(N,) array of atomic masses from `mass_ref`."""
        if self._masses is None:
            self._masses = np.array(
                [mass_ref[lbl] for lbl in self.atomlbls], dtype=np.float32
            )
        return self._masses

    @property
    def com(self) -> np.ndarray:
        """Center of mass of the residue (3,)."""
        if self._com is None:
            self._com = np.average(self.crds, axis=0, weights=self.masses)
        return self._com

    # ------------------------------------------------------------------
    # orientation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def rmat_2_quat(rmat: np.ndarray) -> np.ndarray:
        """Convert a rotation matrix (3x3) to a quaternion (x, y, z, w)."""
        return Rot.from_matrix(rmat).as_quat()

    def mw_pca_axes(
        self,
        n_comps: int = 3,
        rot: str = "rmat",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Mass-weighted PCA of this residue.

        Parameters
        ----------
        n_comps
            Number of PCA components to compute (2 or 3).
        rot
            'rmat' → return (axes, eigenvalues, rotation_matrix)
            'quat' → return (axes, eigenvalues, quaternion_xyzw)

        Returns
        -------
        axes
            (3, 3) array: rows are principal axes (long, short, normal).
        explained_variance
            (3,) eigenvalues.
        rot_rep
            Either 3x3 rotation matrix or quaternion (x, y, z, w).
        """
        if rot not in ("rmat", "quat"):
            raise ValueError(" ## rot must be set to 'rmat' or 'quat'")

        x = self.crds - self.com  # center on COM
        xw = x * np.sqrt(self.masses)[:, None]  # mass-weighting

        pca = PCA(n_components=n_comps, svd_solver="full").fit(xw)
        axes = pca.components_.copy()  # rows

        # If only 2 components requested, construct a normal via cross product
        if n_comps == 2:
            normal = np.cross(axes[0], axes[1])
            normal /= np.linalg.norm(normal)
            axes = np.vstack([axes, normal])

        rmat = axes.T  # columns are axes
        # Ensure right-handed basis
        if np.linalg.det(rmat) < 0:
            axes[2] *= -1.0
            rmat = axes.T

        if rot == "rmat":
            return axes, pca.explained_variance_, rmat

        # quaternion branch
        q_xyzw = self.rmat_2_quat(rmat)
        # enforce unique sign convention
        if q_xyzw[3] < 0:
            q_xyzw = -q_xyzw
        return axes, pca.explained_variance_, q_xyzw
