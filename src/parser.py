# parser.py
import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="structural_analysis_GB",
        description="GB multi-slab workflow (Steps 1–3: slab extraction, grain/GB classification, GB Type layer selection).",
    )

    # ---- General I/O ----
    p.add_argument(
        "-i",
        "--input-gro",
        dest="in_gro",
        default=None,
        help="Input full-system .gro file",
    )
    p.add_argument(
        "--resname",
        default="en-",
        help="Residue name of molecules to analyze",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        dest="out_dir",
        default="results",
        help="Output directory for all slabs and results",
    )
    p.add_argument(
        "--out-basename",
        dest="out_basename",
        default="gb",
        help="Base name used for slab files (default: gb)",
    )

    # ---- Step 1 config ----
    p.add_argument("--thickness-A", type=float, default=80.0, help="Slab thickness (Å) in Step 1")
    p.add_argument("--neigh-cutoff", type=float, default=5.0, help="Neighbor cutoff for smoothing labels (Å)")
    p.add_argument("--smooth-iters", type=int, default=3, help="Number of smoothing iterations (Step 1)")
    p.add_argument("--dbscan-eps", type=float, default=10.0, help="DBSCAN eps for GB clustering (Å)")
    p.add_argument("--dbscan-min-samples", type=int, default=6, help="DBSCAN min_samples for GB clustering")
    p.add_argument(
        "--select-by",
        choices=["resid", "atom"],
        default="resid",
        help="Select slab atoms by 'resid' or 'atom' distances to GB plane",
    )
    p.add_argument(
        "--unwrap",
        action="store_true",
        help="Apply MDAnalysis unwrap transformation before analysis",
    )
    p.add_argument(
        "--no-step1-summary",
        dest="write_step1_summary",
        action="store_false",
        help="Do not write summary_step1_slabs.txt",
    )
    p.set_defaults(write_step1_summary=True)

    # ---- Step 2 config ----
    p.add_argument("--th-high", type=float, default=0.97, help="Orientation threshold TH_HIGH")
    p.add_argument("--margin", type=float, default=0.08, help="Orientation margin for grain assignment")
    p.add_argument("--smooth-iters-2", type=int, default=3, help="Smoothing iterations for Step 2 labels")
    p.add_argument("--connect-radius", type=float, default=7.5, help="Radius for connectivity (Å)")
    p.add_argument("--edge-radius", type=float, default=7.5, help="Radius for edge neighbor detection (Å)")
    p.add_argument("--opp-min-nb", type=int, default=2, help="Min neighbors from opposite grain to mark boundary")
    p.add_argument("--edge-dilate-steps", type=int, default=1, help="GB dilatation steps")
    p.add_argument("--min-gb-size", type=int, default=0, help="Minimum GB component size to keep")
    p.add_argument(
        "--out-prefix-per-slab",
        dest="out_prefix_per_slab",
        default="resids",
        help="Prefix for Step 2 resid .txt files",
    )

    # ---- Step 3 config ----
    p.add_argument(
        "--gb-axis",
        dest="gb_axis",
        choices=["x", "y", "z"],
        default="y",
        help="Axis along which GB is oriented (default: y)",
    )
    p.add_argument("--a-len", type=float, default=6.2753, help="Unit-cell length a (Å)")
    p.add_argument("--b-len", type=float, default=7.7138, help="Unit-cell length b (Å)")
    p.add_argument("--c-len", type=float, default=14.4424, help="Unit-cell length c (Å)")
    p.add_argument("--alpha", type=float, default=76.75, help="Unit-cell angle alpha (deg)")
    p.add_argument("--beta", type=float, default=88.01, help="Unit-cell angle beta (deg)")
    p.add_argument("--gamma", type=float, default=84.52, help="Unit-cell angle gamma (deg)")

    p.add_argument("--slab-thick", type=float, default=40.0, help="Thickness of layer window (Å)")
    p.add_argument("--gb-offset", type=float, default=15.0, help="Distance from GB center to grain bands (Å)")
    p.add_argument("--gb-band-thick", type=float, default=30.0, help="Thickness of GB band (Å)")
    p.add_argument("--box-margin", type=float, default=20.0, help="Margin along third axis to avoid edges (Å)")
    p.add_argument("--connect-radius-3", type=float, default=6.0, help="Radius for Step 3 connectivity (Å)")

    p.add_argument(
        "--step3-mode",
        choices=["single", "multi", "ask"],
        default="single",
        help="Step 3 mode: single mid-layer / 3-layer merge / ask interactively",
    )
    p.add_argument(
        "--merge-tol-y",
        type=float,
        default=5.0,
        help="Merge tolerance (Å) for y0 across layers (multi-layer mode)",
    )

    # thresholds
    p.add_argument(
        "--min-count-to-write",
        type=int,
        default=40,
        help="Min GB/G1/G2 count to write merged GRO/TXT in multi-layer mode",
    )
    p.add_argument(
        "--min-gb-to-write",
        type=int,
        default=40,
        help="Min GB count to write single-layer GRO/TXT (old single-layer behavior)",
    )
    p.add_argument(
        "--target-per-side",
        type=int,
        default=300,
        help="Cap per-grain count in single-layer mode (0 to disable capping)",
    )

    # ---- Outputs control (Step 3 + final) ----
    p.add_argument(
        "--no-step3-gro",
        dest="write_step3_gro",
        action="store_false",
        help="Do not write *_g1/_g2/_gb.gro from Step 3",
    )
    p.add_argument(
        "--no-step3-txt",
        dest="write_step3_txt",
        action="store_false",
        help="Do not write *_g1/_g2/_gb.txt from Step 3",
    )
    p.set_defaults(write_step3_gro=True, write_step3_txt=False)

    p.add_argument(
        "--final-summary-name",
        dest="final_summary_name",
        default="FINAL_gb_summary.txt",
        help="Name of final summary file (Step 3)",
    )
    p.add_argument(
        "--no-final-summary",
        dest="write_final_summary",
        action="store_false",
        help="Do not write final summary file",
    )
    p.set_defaults(write_final_summary=True)
    
        # ---- Latvecs (lattice vector) analysis ----
    p.add_argument(
        "--latvecs",
        dest="do_latvecs",
        action="store_true",
        help=(
            "Run lattice-vector analysis (latvecs) for each grain *_g1/_g2.gro "
            "written in Step 3. Creates <stem>_output.txt next to each grain .gro."
        ),
    )
    p.set_defaults(do_latvecs=False)

    p.add_argument(
        "--latvecs-lam",
        type=float,
        default=4.0,
        help="λ parameter for feature weighting in latvec clustering (default: 4.0)",
    )
    p.add_argument(
        "--latvecs-eps",
        type=float,
        default=0.6,
        help="DBSCAN eps for latvec clustering (default: 0.6)",
    )
    p.add_argument(
        "--latvecs-min-samples",
        type=int,
        default=5,
        help="DBSCAN min_samples for latvec clustering (default: 5)",
    )
    p.add_argument(
        "--latvecs-top-k",
        type=int,
        default=5,
        help="Number of shortest-distance clusters inspected in latvec analysis (default: 5)",
    )
    p.add_argument(
        "--latvecs-visualize",
        action="store_true",
        help="Visualize COMs with py3Dmol during latvec analysis (if installed).",
    )
        # ---- Contact-plane analysis (requires latvecs outputs) ----
    p.add_argument(
        "--contactplanes",
        dest="do_contactplanes",
        action="store_true",
        help=(
            "Run contact-plane analysis for each group using grain COMs and "
            "ff/ef vectors from latvecs outputs (<stem>_output.txt)."
        ),
    )
    p.set_defaults(do_contactplanes=False)

    # ---- Optional misorientation analysis (depends on contactplanes + latvecs) ----
    p.add_argument(
        "--misori",
        dest="do_misori",
        action="store_true",
        help=(
            "Also compute misorientation (Θ, twist, tilt) via orix for groups "
            "where contact-plane analysis succeeds."
        ),
    )
    p.set_defaults(do_misori=False)

    p.add_argument(
        "--misori_symmetry",
        default="triclinic",
        help=(
            "Crystal symmetry for misorientation (e.g. 'triclinic', '2/m', 'mmm'). "
            "Default: 'triclinic'."
        ),
    )
        # ---- Optional filtering of final summary by contact_plane pattern ----
    p.add_argument(
        "--filter-planes",
        dest="do_filter_planes",
        action="store_true",
        help=(
            "After writing FINAL summary, also write a filtered 'best-per-rank' "
            "file selecting one group per slab_rank that matches a given "
            "contact_plane pattern (e.g. 'ac-ac')."
        ),
    )
    p.set_defaults(do_filter_planes=False)

    p.add_argument(
        "--filter_plane_pair",
        default="ac-ac",
        help=(
            "Desired contact_plane pattern as 'g1plane-g2plane', "
            "e.g. 'ac-ac', 'ab-ac', 'ac-*'. "
            "Default: 'ac-ac'."
        ),
    )

    p.add_argument(
        "--filter_out_name",
        default="FILTERED_best_per_rank.txt",
        help=(
            "Name of the filtered summary TSV to write in out_dir "
            "(default: FILTERED_best_per_rank.txt)."
        ),
    )

    return p


