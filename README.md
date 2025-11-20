# Structural Analysis of Grain Boundaries (GB)

A Python package for automated structural analysis of grain boundaries in molecular systems. This package processes molecular dynamics simulation output files (GROMACS `.gro` format) to identify, classify, and analyze grain boundaries in crystalline materials.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow Steps](#workflow-steps)
- [Usage Examples](#usage-examples)
- [Command-Line Options](#command-line-options)
- [Output Files](#output-files)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Overview

The `structural_analysis_GB` package implements a multi-step workflow for grain boundary analysis:

1. **Step 1**: Extract multiple slabs from a full-system structure
2. **Step 2**: Classify residues/molecules into grains and grain boundaries
3. **Step 3**: Select representative layers and output grain/GB segments
4. **Step 4** (Optional): Advanced analysis including lattice vectors, contact planes, and misorientation angles

## Features

### Core Capabilities

- **Automated Slab Extraction**: Detect and extract multiple grain boundary regions from large simulation boxes
- **Grain Segmentation**: Classify molecules into two distinct grains (G1, G2) and grain boundary (GB) regions
- **Layer Selection**: Extract single or multi-layer representations of grain boundaries
- **Connectivity Analysis**: Identify connected components within grains and boundaries

### Advanced Analysis

- **Lattice Vector Analysis**: Determine crystallographic orientations of individual grains
- **Contact Plane Detection**: Identify crystallographic planes at the grain boundary interface
- **Misorientation Calculation**: Compute crystallographic misorientation angles (requires `orix`)
- **Plane Filtering**: Select best grain boundaries based on contact plane criteria

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/sonaligarg1807/structural_analysis_GB.git
cd structural_analysis_GB

# Install required dependencies
pip install -r requirements.txt
```

### Development Installation

For development work with additional tools (linting, testing, type checking):

```bash
pip install -e ".[dev]"
```

### Optional Dependencies

For visualization capabilities:

```bash
pip install ".[visual]"  # Adds py3Dmol for 3D visualization
```

For misorientation analysis:

```bash
pip install orix  # Required for crystallographic misorientation calculations
```

## Quick Start

### Basic Example

Process a GROMACS `.gro` file to extract and analyze grain boundaries:

```bash
python -m main \
    --input-gro your_structure.gro \
    --resname PEN \
    --out-dir output_results \
    --step3-mode single
```

This will:
- Extract all grain boundary slabs from `your_structure.gro`
- Classify molecules with residue name "PEN" into grains and boundaries
- Output single-layer grain/GB structures
- Generate a summary file with all results

### Complete Analysis with Advanced Features

```bash
python -m main \
    --input-gro your_structure.gro \
    --resname PEN \
    --out-dir output_results \
    --step3-mode multi \
    --latvecs \
    --contactplanes \
    --misori \
    --filter-planes
```

This runs the full pipeline including lattice vector analysis, contact plane detection, and misorientation calculations.

## Workflow Steps

### Step 1: Slab Extraction

**Purpose**: Identify and extract grain boundary regions from the input structure.

**How it works**:
1. Computes center-of-mass (COM) and principal normal vectors for each molecule
2. Uses 2-means clustering on molecular orientations to find grain boundary planes
3. Applies DBSCAN clustering to group nearby GB points
4. Extracts slab regions around each identified GB with specified thickness

**Key Parameters**:
- `--thickness-A`: Thickness of extracted slabs in Ångströms (default: 80.0)
- `--dbscan-eps`: Clustering distance for GB detection (default: 10.0)
- `--select-by`: Select by 'resid' (whole molecules) or 'atom' (individual atoms)

**Outputs**:
- `slab0.gro`, `slab1.gro`, ... : Individual slab structure files
- `summary_step1_slabs.txt`: Summary of all extracted slabs

### Step 2: Grain/GB Segmentation

**Purpose**: Classify molecules in each slab into two grains (G1, G2) and grain boundary (GB).

**How it works**:
1. Computes molecular orientations (gyration tensor long axis)
2. Uses nematic embedding and K-means clustering to identify two grain orientations
3. Classifies molecules based on orientation similarity to grain directors
4. Identifies GB molecules that have neighbors from both grains
5. Applies spatial smoothing to refine classifications

**Key Parameters**:
- `--th-high`: Orientation threshold for grain assignment (default: 0.97)
- `--margin`: Orientation margin for grain assignment (default: 0.08)
- `--connect-radius`: Radius for connectivity analysis (default: 7.5 Å)
- `--edge-radius`: Radius for boundary detection (default: 7.5 Å)

**Outputs**:
- `resids_grain1.txt`, `resids_grain2.txt`, `resids_GB.txt`: Residue IDs for each class
- Classification metadata in JSON format

### Step 3: Layer Selection

**Purpose**: Select representative molecular layers from the classified grains and boundaries.

**Modes**:

#### Single-Layer Mode (`--step3-mode single`)
Extracts molecules from the middle layer perpendicular to the GB axis.

**Use when**: You want a thin, representative slice through the grain boundary.

**Key Parameters**:
- `--gb-axis`: Axis perpendicular to GB plane (x, y, or z; default: y). This is the axis along which layers are selected.
- `--slab-thick`: Thickness of selection window (default: 40.0 Å)
- `--min-gb-to-write`: Minimum GB count to output files (default: 40)
- `--target-per-side`: Cap molecules per grain (default: 300, 0 to disable)

#### Multi-Layer Mode (`--step3-mode multi`)
Extracts and merges molecules from three layers (top, middle, bottom).

**Use when**: You want more comprehensive sampling across the GB region.

**Key Parameters**:
- `--merge-tol-y`: Tolerance for merging groups across layers (default: 5.0 Å)
- `--min-count-to-write`: Minimum molecule count per region (GB, G1, or G2) required to write output files (default: 40)

#### Interactive Mode (`--step3-mode ask`)
Prompts for mode selection for each slab during execution.

**Outputs**:
- `*_g1.gro`, `*_g2.gro`: Grain 1 and Grain 2 structure files
- `*_gb.gro`: Grain boundary structure file
- `*_g1.txt`, `*_g2.txt`, `*_gb.txt`: Residue ID lists (optional)
- `FINAL_gb_summary.txt`: Comprehensive summary of all groups

### Step 4: Advanced Analysis (Optional)

#### Lattice Vector Analysis (`--latvecs`)

**Purpose**: Determine crystallographic lattice vectors for each grain.

**How it works**:
1. Computes pairwise distances and vectors between molecular centers
2. Clusters vectors using DBSCAN with orientation-weighted features
3. Identifies face-face (ff) and edge-face (ef) lattice vectors
4. Outputs lattice parameters and orientations

**Key Parameters**:
- `--latvecs-lam`: Feature weighting parameter λ (default: 4.0)
- `--latvecs-eps`: DBSCAN clustering epsilon (default: 0.6)
- `--latvecs-min-samples`: Minimum samples per cluster (default: 5)
- `--latvecs-top-k`: Number of clusters to inspect (default: 5)
- `--latvecs-visualize`: Enable 3D visualization (requires py3Dmol)

**Outputs**:
- `*_output.txt`: Lattice vectors and distances for each grain

#### Contact Plane Analysis (`--contactplanes`)

**Purpose**: Identify crystallographic planes at the grain boundary interface.

**Requirements**: Must run with `--latvecs` first.

**How it works**:
1. Reads lattice vectors from latvecs output
2. Computes grain centers of mass
3. Determines contact vector between grains
4. Projects contact vector onto lattice planes (a, b, c)
5. Reports dominant contact planes (e.g., "ac-ac" for both grains on ac plane)

**Outputs**:
- Contact plane labels added to `FINAL_gb_summary.txt`

#### Misorientation Analysis (`--misori`)

**Purpose**: Calculate crystallographic misorientation between grains.

**Requirements**: 
- Must run with `--latvecs` and `--contactplanes`
- Requires `orix` package installation

**How it works**:
1. Constructs rotation matrices from lattice vectors
2. Uses `orix` library for crystallographic calculations
3. Computes misorientation angle (Θ), twist, and tilt components
4. Accounts for crystal symmetry

**Key Parameters**:
- `--misori_symmetry`: Crystal symmetry for misorientation calculations (default: "triclinic")
  - Common options: 
    - "triclinic" or "-1": Triclinic with inversion center (for pentacene, general organic crystals)
    - "2/m": Monoclinic with 2-fold rotation and mirror
    - "mmm": Orthorhombic with three perpendicular mirror planes
  - The symmetry affects how equivalent crystallographic orientations are identified
  - Use the symmetry that matches your crystal system for accurate misorientation calculations

**Outputs**:
- `misori_deg`, `twist_deg`, `tilt_deg` columns in `FINAL_gb_summary.txt`

#### Plane Filtering (`--filter-planes`)

**Purpose**: Select best grain boundaries matching specific contact plane criteria.

**How it works**:
1. Filters results to match specified contact plane pattern
2. Selects one best group per slab based on:
   - Minimal distance to box center
   - Maximal grain balance
   - Maximal total grain size
   - Maximal GB size

**Key Parameters**:
- `--filter_plane_pair`: Desired pattern (default: "ac-ac")
  - Examples: "ac-ac", "ab-ab", "ac-*" (wildcard for grain 2)
- `--filter_out_name`: Output filename (default: "ac-ac_best.txt")

**Outputs**:
- Filtered summary file with best matches

## Usage Examples

### Example 1: Basic Analysis

Extract grain boundaries and perform basic segmentation:

```bash
python -m main \
    --input-gro system.gro \
    --resname PEN \
    --out-dir results_basic \
    --step3-mode single
```

### Example 2: Custom Unit Cell Parameters

For materials with specific unit cell dimensions:

```bash
python -m main \
    --input-gro system.gro \
    --resname MOL \
    --out-dir results_custom \
    --a-len 7.5 \
    --b-len 9.0 \
    --c-len 12.5 \
    --alpha 90 \
    --beta 95 \
    --gamma 90 \
    --step3-mode multi
```

### Example 3: Complete Crystallographic Analysis

Full analysis with lattice vectors, contact planes, and misorientation:

```bash
python -m main \
    --input-gro pentacene.gro \
    --resname PEN \
    --out-dir results_full \
    --step3-mode multi \
    --latvecs \
    --contactplanes \
    --misori \
    --misori_symmetry triclinic \
    --filter-planes \
    --filter_plane_pair ac-ac
```

### Example 4: High-Precision Segmentation

Fine-tune segmentation parameters for challenging systems:

```bash
python -m main \
    --input-gro complex_system.gro \
    --resname ORG \
    --out-dir results_precise \
    --th-high 0.98 \
    --margin 0.05 \
    --connect-radius 6.0 \
    --smooth-iters 5 \
    --smooth-iters-2 5 \
    --step3-mode single
```

### Example 5: Unwrapped Coordinates

For systems requiring unwrapping of periodic boundary conditions:

```bash
python -m main \
    --input-gro wrapped_system.gro \
    --resname MOL \
    --out-dir results_unwrapped \
    --unwrap \
    --step3-mode multi
```

### Example 6: Interactive Mode

For hands-on control of layer selection:

```bash
python -m main \
    --input-gro system.gro \
    --resname PEN \
    --out-dir results_interactive \
    --step3-mode ask
```

When prompted, enter 'm' for multi-layer or 's' for single-layer for each slab.

## Command-Line Options

### General I/O Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-i, --input-gro` | str | - | Input GROMACS .gro file (required) |
| `--resname` | str | "en-" | Residue name of molecules to analyze |
| `-o, --out-dir` | str | "results" | Output directory for all results |
| `--out-basename` | str | "gb" | Base name for slab files |

### Step 1: Slab Extraction Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--thickness-A` | float | 80.0 | Slab thickness in Ångströms |
| `--neigh-cutoff` | float | 5.0 | Neighbor cutoff for smoothing (Å) |
| `--smooth-iters` | int | 3 | Smoothing iterations |
| `--dbscan-eps` | float | 10.0 | DBSCAN epsilon for GB clustering (Å) |
| `--dbscan-min-samples` | int | 6 | DBSCAN minimum samples |
| `--select-by` | choice | "resid" | Selection mode: "resid" or "atom" |
| `--unwrap` | flag | False | Apply MDAnalysis unwrap transformation |
| `--no-step1-summary` | flag | - | Disable Step 1 summary output |

### Step 2: Grain/GB Segmentation Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--th-high` | float | 0.97 | Orientation threshold for grain assignment |
| `--margin` | float | 0.08 | Orientation margin |
| `--smooth-iters-2` | int | 3 | Smoothing iterations for labels |
| `--connect-radius` | float | 7.5 | Connectivity radius (Å) |
| `--edge-radius` | float | 7.5 | Edge neighbor detection radius (Å) |
| `--opp-min-nb` | int | 2 | Min opposite-grain neighbors for GB |
| `--edge-dilate-steps` | int | 1 | GB dilation steps |
| `--min-gb-size` | int | 0 | Minimum GB component size |
| `--out-prefix-per-slab` | str | "resids" | Prefix for Step 2 output files |

### Step 3: Layer Selection Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--gb-axis` | choice | "y" | Axis perpendicular to the GB plane: x, y, or z |
| `--a-len` | float | 6.2753 | Unit cell length a (Å) |
| `--b-len` | float | 7.7138 | Unit cell length b (Å) |
| `--c-len` | float | 14.4424 | Unit cell length c (Å) |
| `--alpha` | float | 76.75 | Unit cell angle alpha (degrees) |
| `--beta` | float | 88.01 | Unit cell angle beta (degrees) |
| `--gamma` | float | 84.52 | Unit cell angle gamma (degrees) |
| `--slab-thick` | float | 40.0 | Layer window thickness (Å) |
| `--gb-offset` | float | 15.0 | Distance from GB center to grain bands (Å) |
| `--gb-band-thick` | float | 30.0 | GB band thickness (Å) |
| `--box-margin` | float | 20.0 | Margin to avoid box edges (Å) |
| `--connect-radius-3` | float | 6.0 | Step 3 connectivity radius (Å) |
| `--step3-mode` | choice | "single" | Mode: "single", "multi", or "ask" |
| `--merge-tol-y` | float | 5.0 | Merge tolerance for multi-layer (Å) |
| `--min-count-to-write` | int | 40 | Min count for multi-layer output |
| `--min-gb-to-write` | int | 40 | Min GB count for single-layer output |
| `--target-per-side` | int | 300 | Per-grain count cap (0=disable) |

### Step 3: Output Control Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--no-step3-gro` | flag | - | Disable .gro output from Step 3 |
| `--no-step3-txt` | flag | - | Disable .txt output from Step 3 |
| `--final-summary-name` | str | "FINAL_gb_summary.txt" | Final summary filename |
| `--no-final-summary` | flag | - | Disable final summary output |

### Step 4: Lattice Vector Analysis Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--latvecs` | flag | False | Enable lattice vector analysis |
| `--latvecs-lam` | float | 4.0 | Feature weighting parameter λ |
| `--latvecs-eps` | float | 0.6 | DBSCAN epsilon for clustering |
| `--latvecs-min-samples` | int | 5 | DBSCAN minimum samples |
| `--latvecs-top-k` | int | 5 | Number of clusters to inspect |
| `--latvecs-visualize` | flag | False | Enable 3D visualization |

### Step 4: Contact Plane Analysis Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--contactplanes` | flag | False | Enable contact plane analysis |

### Step 4: Misorientation Analysis Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--misori` | flag | False | Enable misorientation calculation |
| `--misori_symmetry` | str | "triclinic" | Crystal symmetry for misorientation |

### Step 4: Filtering Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--filter-planes` | flag | False | Enable contact plane filtering |
| `--filter_plane_pair` | str | "ac-ac" | Desired contact plane pattern |
| `--filter_out_name` | str | "ac-ac_best.txt" | Filtered output filename |

## Output Files

### Directory Structure

After running the pipeline, the output directory will contain:

```
output_results/
├── summary_step1_slabs.txt          # Step 1 summary
├── slab0/
│   ├── slab0.gro                    # Extracted slab structure
│   ├── resids_grain1.txt            # Grain 1 residue IDs
│   ├── resids_grain2.txt            # Grain 2 residue IDs
│   ├── resids_GB.txt                # GB residue IDs
│   ├── slab0_mid_g1.gro             # Grain 1 structure
│   ├── slab0_mid_g2.gro             # Grain 2 structure
│   ├── slab0_mid_gb.gro             # GB structure
│   ├── slab0_mid_g1_output.txt      # Grain 1 lattice vectors (if --latvecs)
│   └── slab0_mid_g2_output.txt      # Grain 2 lattice vectors (if --latvecs)
├── slab1/
│   └── ...
└── FINAL_gb_summary.txt             # Comprehensive final summary
```

### File Formats

#### `summary_step1_slabs.txt`

Tab-separated values with columns:
- `slab_idx`: Slab index
- `slab_file`: Path to slab .gro file
- `n_atoms`: Number of atoms in slab
- `gb_plane_normal`: Normal vector to GB plane
- `gb_center`: Center position of GB

#### `resids_*.txt`

Plain text files with one residue ID per line.

#### `*_output.txt` (Lattice Vectors)

Format:
```
# idx type distance vec_x vec_y vec_z
0 ff 5.234 1.234 2.345 3.456
1 ef 7.891 -0.567 0.890 1.234
```

Where:
- `ff`: Face-face lattice vector
- `ef`: Edge-face lattice vector

#### `FINAL_gb_summary.txt`

Tab-separated values with columns:
- `slab_rank`: Slab ranking
- `slab_dir`: Directory containing slab
- `slab_stem`: Slab base name
- `GB_AXIS`: Grain boundary axis (x, y, or z)
- `y_center_A`: Center position along GB axis (Å)
- `dist_to_box_center_A`: Distance to box center (Å)
- `GB_N`: Number of molecules in GB region
- `G1_N`: Number of molecules in Grain 1
- `G2_N`: Number of molecules in Grain 2
- `contact_plane`: Contact plane labels (e.g., "g1=ac,g2=ac")
- `misori_deg`: Misorientation angle (degrees, if calculated)
- `twist_deg`: Twist component (degrees, if calculated)
- `tilt_deg`: Tilt component (degrees, if calculated)

## Advanced Features

### Customizing Unit Cell Parameters

The package uses default unit cell parameters for pentacene. For other materials, specify custom parameters:

```bash
python -m main \
    --input-gro your_material.gro \
    --resname YOUR \
    --a-len 5.0 \
    --b-len 6.0 \
    --c-len 10.0 \
    --alpha 90 \
    --beta 90 \
    --gamma 90
```

### Working with Different GB Axes

If grain boundaries are oriented along different axes:

```bash
# For GB perpendicular to x-axis
python -m main --input-gro system.gro --gb-axis x ...

# For GB perpendicular to z-axis
python -m main --input-gro system.gro --gb-axis z ...
```

### Optimizing for Large Systems

For very large systems, consider:

1. Increase `--thickness-A` to capture full GB regions
2. Adjust `--dbscan-eps` for better GB detection
3. Use `--select-by atom` for finer control
4. Increase `--connect-radius` for better connectivity

### Selective Visualization

Enable visualization only during latvec analysis:

```bash
python -m main \
    --input-gro system.gro \
    --resname PEN \
    --latvecs \
    --latvecs-visualize
```

Requires installation: `pip install ".[visual]"`

## Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError: No module named 'numpy'`

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

---

**Problem**: No slabs extracted (Step 1 finds no grain boundaries)

**Solutions**:
- Adjust `--dbscan-eps` (try larger values like 15.0 or 20.0)
- Verify `--resname` matches your molecule names
- Check input file contains grain boundary structures
- Try `--dbscan-min-samples` with lower values (e.g., 3 or 4)

---

**Problem**: Misorientation calculation fails

**Solutions**:
- Install orix: `pip install orix`
- Ensure `--latvecs` and `--contactplanes` are enabled
- Verify lattice vector outputs exist
- Check crystal symmetry setting matches your material

---

**Problem**: No output .gro files generated in Step 3

**Solutions**:
- Lower `--min-gb-to-write` threshold (try 20 or 10)
- Lower `--min-count-to-write` for multi-layer mode
- Check that grain/GB regions are large enough
- Verify Step 2 successfully classified molecules

---

**Problem**: Grain classification looks incorrect

**Solutions**:
- Adjust `--th-high` (try 0.95 or 0.98)
- Modify `--margin` (try 0.05 or 0.10)
- Increase `--smooth-iters-2` (try 5 or 7)
- Check that molecules have clear orientational order

---

**Problem**: Memory issues with large systems

**Solutions**:
- Process slabs individually if possible
- Reduce `--thickness-A` to extract smaller slabs
- Use `--select-by resid` instead of `atom`
- Consider splitting input file into smaller regions

## Citation

If you use this package in your research, please cite:

```bibtex
@software{structural_analysis_gb,
  author = {Garg, Sonali},
  title = {Structural Analysis of Grain Boundaries},
  year = {2024},
  url = {https://github.com/sonaligarg1807/structural_analysis_GB}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/sonaligarg1807/structural_analysis_GB/issues).
