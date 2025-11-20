"""
structural_analysis_gb package initializer.

This file keeps imports light-weight (no heavy imports at package import time)
and exposes package metadata and the public module list.

Note:
- Do not import modules that require heavy C/Python extensions or that perform
  I/O at import time. Import those modules inside functions when needed.
"""
__all__ = [
    "parser",
    "utils",
    "slab_extr",
    "grain_gb_segmentation",
    "GB_type_segm",
    "latvecs",
    "contactplanes",
    "misorientation_angle",
    "filter_planes",
    "grotools",
    "iofile",
    "site",
]

# package version (bump when releasing)
__version__ = "0.1.0"

# Friendly short description
__description__ = "Structural grain-boundary analysis utilities (slab extraction, GB segmentation, layer selection, latvec/contact-plane/misorientation analysis)."