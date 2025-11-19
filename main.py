#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GB multi-slab workflow (package version)
Steps:
  1) Step-1: extract slabs + metadata
  2) Step-2: topological grain/GB classification (writes resid TXT)
  3) Step-3: layer selection (single- or multi-layer) and (optionally) write *_g1/_g2/_gb.gro per segment/group
  4) (optional) Per-grain lattice-vector analysis + contact-plane labels for groups that have *_g1/_g2.gro

User typically runs:
  python -m src.main \\
      --in-gro b45.gro \\
      --out-dir b45_gb_workflow_out \\
      --resname PEN \\
      --step3-mode multi \\
      --do-latvecs --do-contactplanes
"""

import os
import re
from pathlib import Path

from src.parser import build_parser
from src.slab_extr import step1_extract_all_slabs
from src.grain_gb_segmentation import step2_clustering_for_slab
from src.GB_type_segm import (
    step3_single_layer_for_slab,
    step3_merge_layers_for_slab,
)
from src.latvecs import analyze_grain_latvecs
from src.contactplanes import contactplanes_for_group
from src.misorientation_angle import misorientation_for_group
from src.filter_planes import filter_best_per_rank


def run(args=None):
    parser = build_parser()
    args = parser.parse_args(args)

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Step 1: extract all slabs from the input .gro ----------
    slabs = step1_extract_all_slabs(
        in_gro=args.in_gro,
        resname=args.resname,
        out_dir=args.out_dir,
        out_base=args.out_basename,
        thickness_A=args.thickness_A,
        neigh_cutoff=args.neigh_cutoff,
        smooth_iters=args.smooth_iters,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        select_by=args.select_by,
        unwrap=args.unwrap,
        write_summary=args.write_step1_summary,
    )

    # ---------- Step 2 + Step 3 over all slabs ----------
    final_rows = []
    for slab in slabs:
        slab_dir = Path(slab).parent

        step2_out = step2_clustering_for_slab(
            slab_gro=slab,
            slab_dir=slab_dir,
            out_prefix=args.out_prefix_per_slab,
            th_high=args.th_high,
            margin=args.margin,
            smooth_iters=args.smooth_iters_2,
            connect_radius=args.connect_radius,
            edge_radius=args.edge_radius,
            opp_min_nb=args.opp_min_nb,
            edge_dilate_steps=args.edge_dilate_steps,
            min_gb_size=args.min_gb_size,
        )

        # --- Step-3 mode selection ---
        mode = args.step3_mode.strip().lower()
        if mode not in ("single", "multi", "ask"):
            print(f"[warn] STEP3_MODE='{args.step3_mode}' not recognized. Using 'ask'.")
            mode = "ask"

        if mode == "ask":
            while True:
                choice = input(
                    f"[step3] For slab '{Path(slab).stem}': choose mode "
                    f"[m]ulti (3-layer) / [s]ingle (mid-layer only) → "
                ).strip().lower()
                if choice in ("m", "multi"):
                    mode_eff = "multi"
                    break
                if choice in ("s", "single"):
                    mode_eff = "single"
                    break
                print("  Please type 'm' or 's'.")
        else:
            mode_eff = mode

        # --- Step-3 proper ---
        if mode_eff == "multi":
            rows = step3_merge_layers_for_slab(
                slab_gro=slab,
                slab_dir=slab_dir,
                step2_out=step2_out,
                gb_axis=args.gb_axis,
                a_len=args.a_len,
                b_len=args.b_len,
                c_len=args.c_len,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                slab_thick=args.slab_thick,
                gb_offset=args.gb_offset,
                gb_band_thick=args.gb_band_thick,
                box_margin=args.box_margin,
                connect_radius_layer=args.connect_radius_3,
                merge_tol_y=args.merge_tol_y,
                write_txt=args.write_step3_txt,
                write_gro=args.write_step3_gro,
                min_count_write=args.min_count_to_write,
            )
        else:
            rows = step3_single_layer_for_slab(
                slab_gro=slab,
                slab_dir=slab_dir,
                step2_out=step2_out,
                gb_axis=args.gb_axis,
                a_len=args.a_len,
                b_len=args.b_len,
                c_len=args.c_len,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                slab_thick=args.slab_thick,
                gb_offset=args.gb_offset,
                gb_band_thick=args.gb_band_thick,
                box_margin=args.box_margin,
                connect_radius_layer=args.connect_radius_3,
                write_txt=args.write_step3_txt,
                write_gro=args.write_step3_gro,
                min_count_write=args.min_count_to_write,
                min_gb_to_write=args.min_gb_to_write,
                target_per_side=(
                    args.target_per_side if args.target_per_side > 0 else None
                ),
            )

        # add slab rank from stem if present
        for r in rows:
            m = re.search(r"rank(\d+)", r["slab_stem"])
            r["slab_rank"] = int(m.group(1)) if m else -1
        final_rows.extend(rows)

    # ---------- Optional Step-4: latvecs + contact-planes per group ----------
    # we do this BEFORE writing the final summary, so contact_plane is up-to-date
    if args.do_latvecs or args.do_contactplanes:
        print(
            "[step4] Scanning groups with *_g1/_g2.gro for latvec/contact-plane analysis..."
        )

        for r in final_rows:
            paths = r.get("paths")
            if not paths:
                continue

            g1_path = paths.get("g1")
            g2_path = paths.get("g2")
            if not (g1_path and g2_path):
                continue

            # --- 4a) Lattice-vector analysis (per grain) ---
            if args.do_latvecs:
                # Grain 1
                try:
                    analyze_grain_latvecs(
                        g1_path,
                        output_txt=None,  # default: <stem>_output.txt in same dir
                        visualize=args.latvecs_visualize,
                        lam=args.latvecs_lam,
                        eps=args.latvecs_eps,
                        min_samples=args.latvecs_min_samples,
                        top_k=args.latvecs_top_k,
                    )
                except Exception as e:
                    print(f"[latvecs] Warning: failed for {g1_path}: {e}")

                # Grain 2
                try:
                    analyze_grain_latvecs(
                        g2_path,
                        output_txt=None,
                        visualize=args.latvecs_visualize,
                        lam=args.latvecs_lam,
                        eps=args.latvecs_eps,
                        min_samples=args.latvecs_min_samples,
                        top_k=args.latvecs_top_k,
                    )
                except Exception as e:
                    print(f"[latvecs] Warning: failed for {g2_path}: {e}")

            # --- 4b) Contact-plane analysis (needs latvec outputs on disk) ---
            if args.do_contactplanes:
                try:
                    cp1, cp2 = contactplanes_for_group(
                        g1_gro_file=g1_path,
                        g2_gro_file=g2_path,
                        g1_txt=None,  # use <stem>_output.txt by default
                        g2_txt=None,
                    )
                    if cp1 is not None and cp2 is not None:
                        # Store combined info in the single contact_plane column
                        r["contact_plane"] = f"g1={cp1},g2={cp2}"

                        # --- Optional misorientation (only if contact-planes succeeded) ---
                        if args.do_misori:
                            try:
                                miso = misorientation_for_group(
                                    g1_gro_file=g1_path,
                                    g2_gro_file=g2_path,
                                    g1_txt=None,  # use <stem>_output.txt by default
                                    g2_txt=None,
                                    symmetry_name=args.misori_symmetry,
                                )
                                r["misori_deg"] = miso["theta_deg"]
                                r["twist_deg"] = miso["twist_deg"]
                                r["tilt_deg"] = miso["tilt_deg"]
                            except Exception as e_m:
                                print(
                                    f"[misori] Warning: failed for {g1_path}, {g2_path}: {e_m}"
                                )
                    else:
                        # no valid contact plane → no misorientation
                        r.setdefault("misori_deg", None)
                        r.setdefault("twist_deg", None)
                        r.setdefault("tilt_deg", None)

                except Exception as e:
                    print(
                        f"[contactplanes] Warning: failed for {g1_path}, {g2_path}: {e}"
                    )
                    r.setdefault("misori_deg", None)
                    r.setdefault("twist_deg", None)
                    r.setdefault("tilt_deg", None)

    # ---------- Final summary (after Step 4 so contact_plane is updated) ----------
    final_rows.sort(key=lambda d: (d["dist_to_box_center_A"], d["slab_rank"]))

    if args.write_final_summary:
        final_path = Path(args.out_dir) / args.final_summary_name
        header = "\t".join(
            [
                "slab_rank",
                "slab_dir",
                "slab_stem",
                "GB_AXIS",
                "y_center_A",
                "dist_to_box_center_A",
                "GB_N",
                "G1_N",
                "G2_N",
                "contact_plane",
                "misori_deg",
                "twist_deg",
                "tilt_deg",
            ]
        )

        def _fmt_float_or_na(x):
            if x is None:
                return "NA"
            try:
                return f"{float(x):.2f}"
            except Exception:
                return "NA"

        with open(final_path, "w") as f:
            f.write(header + "\n")
            for r in final_rows:
                misori = r.get("misori_deg", None)
                twist = r.get("twist_deg", None)
                tilt = r.get("tilt_deg", None)

                f.write(
                    f"{r['slab_rank']}\t"
                    f"{r['slab_dir']}\t"
                    f"{r['slab_stem']}\t"
                    f"{r['gb_axis']}\t"
                    f"{r['y_center_A']:.2f}\t"
                    f"{r['dist_to_box_center_A']:.2f}\t"
                    f"{r['GB_N']}\t"
                    f"{r['G1_N']}\t"
                    f"{r['G2_N']}\t"
                    f"{r['contact_plane']}\t"
                    f"{_fmt_float_or_na(misori)}\t"
                    f"{_fmt_float_or_na(twist)}\t"
                    f"{_fmt_float_or_na(tilt)}\n"
                )

        print(f"[done] Final summary (Step 1–4) → {final_path}")

        # ---- Optional filtering step (best-per-rank by contact_plane pattern) ----
        if getattr(args, "do_filter_planes", False):
            filtered_path = Path(args.out_dir) / args.filter_out_name
            filter_best_per_rank(
                summary_path=final_path,
                out_path=filtered_path,
                plane_pair=args.filter_plane_pair,
            )
    else:
        print("[done] Final summary not written (disabled).")

if __name__ == "__main__":
    run()
