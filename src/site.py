"""Module defining the Site class for molecular residues."""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as Rot
from .grotools import Residue


mass_ref = {"H": 1.001, "C": 12.001}


class Site:
    """Class representing a molecular site (residue) with various properties and methods."""

    def __init__(self, residue: Residue):
        self.residue = residue
        self._com = None
        self._masses = None
        self._crds = None
        self._atomlbls = None
        self._resid = None

    @property
    def resid(self) -> int:
        """Get the residue ID."""
        return self.residue.resid

    @property
    def atomlbls(self) -> list[str]:
        """Get a list of atom names in the residue."""
        atomlbls = [atom.atomname[0] for atom in self.residue.atoms]
        return atomlbls

    @property
    def crds(self) -> np.ndarray:
        """Get the coordinates of all atoms in the residue."""
        crds = np.array(
            [atom.position for atom in self.residue.atoms],
            dtype=np.float32,
        )
        return crds

    @property
    def masses(self) -> np.ndarray:
        """Get the masses of all atoms in the residue."""
        masses = np.array([mass_ref[lbl] for lbl in self.atomlbls], dtype=np.float32)
        return masses

    @property
    def com(self) -> np.ndarray:
        """Calculate and return the center of mass of the residue."""
        com = np.average(self.crds, axis=0, weights=self.masses)
        return com

    @staticmethod
    def rmat_2_quat(rmat: np.ndarray) -> np.ndarray:
        """Convert a rotation matrix to a quaternion."""
        quat = Rot.from_matrix(rmat).as_quat()
        return quat

    def mw_pca_axes(
        self, n_comps: int = 3, rot: str = "rmat"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the principal axes of the residue using mass-weighted PCA."""
        if rot not in ("rmat", "quat"):
            raise ValueError(" ## rot must be set to rmat or quat")

        x = self.crds - self.com
        xw = x * np.sqrt(self.masses)[:, None]  # scale rows by sqrt(m)
        pca = PCA(n_components=n_comps, svd_solver="full").fit(xw)
        axes = pca.components_.copy()
        if n_comps == 2:
            normal = np.cross(axes[0], axes[1])
            normal /= np.linalg.norm(normal)
            axes = np.vstack([axes, normal])
        rmat = axes.T
        if np.linalg.det(rmat) < 0:
            axes[2] *= -1
            rmat = axes.T
        if rot == "rmat":
            return axes, pca.explained_variance_, rmat
        if rot == "quat":
            q_xyzw = self.rmat_2_quat(rmat)
            if q_xyzw[3] < 0:
                q_xyzw = -q_xyzw
            return axes, pca.explained_variance_, q_xyzw
