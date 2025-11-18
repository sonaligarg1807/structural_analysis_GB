#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GB multi-slab workflow (package version)
Steps:
  1) Step-1: extract slabs + metadata
  2) Step-2: topological grain/GB classification (writes resid TXT)
  3) Step-3: layer selection (single- or multi-layer) and (optionally) write *_g1/_g2/_gb.gro per segment/group

NOTE: Contact-plane analysis (Step-4) is *not* included here; you said you have separate code for that.
"""

import os
import re
from pathlib import Path

from structural_analysis_GB.parser import build_parser
from structural_analysis_GB.step1 import step1_extract_all_slabs
from structural_analysis_GB.step2 import step2_clustering_for_slab
from structural_analysis_GB.step3 import (
    step3_single_layer_for_slab,
    step3_merge_layers_for_slab,
)


def run(args=None):
    parser = build_parser()
    args = parser.parse_args(args)

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------- Step 1 ----------
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

    # ---------- Step 2 + Step 3 ----------
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

        # Step-3 mode selection
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
                target_per_side=args.target_per_side if args.target_per_side > 0 else None,
            )

        # add slab rank from stem if present
        for r in rows:
            m = re.search(r"rank(\d+)", r["slab_stem"])
            r["slab_rank"] = int(m.group(1)) if m else -1
        final_rows.extend(rows)

    # ---------- Final summary (up to Step 3) ----------
    # contact_plane stays 'NA' here; your separate Step-4 code can modify this later.
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
            ]
        )
        with open(final_path, "w") as f:
            f.write(header + "\n")
            for r in final_rows:
                f.write(
                    f"{r['slab_rank']}\t{r['slab_dir']}\t{r['slab_stem']}\t{r['gb_axis']}\t"
                    f"{r['y_center_A']:.2f}\t{r['dist_to_box_center_A']:.2f}\t"
                    f"{r['GB_N']}\t{r['G1_N']}\t{r['G2_N']}\t{r['contact_plane']}\n"
                )
        print(f"[done] Final summary (Step 1–3) → {final_path}")
    else:
        print("[done] Final summary not written (disabled).")

    print("Note: contact_plane column is 'NA' here; use your separate Step-4 code to fill it.")


if __name__ == "__main__":
    run()
