#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter the final GB summary to one best group per slab rank.

Selection criteria (applied in order):
  0) KEEP ONLY rows where BOTH grains are 'ac' (or 'ca'):
       contact_plane like 'g1=ac,g2=ac' (case-insensitive, whitespace-tolerant)
  1) minimal dist_to_box_center_A
  2) maximal min(G1_N, G2_N)         (balance: both grains large)
  3) maximal (G1_N + G2_N)           (total grain size)
  4) maximal GB_N                    (size of GB set)

Input  : FINAL_gb_summary.txt produced by the main pipeline
Output : FILTERED_best_per_rank.txt (TSV)

Usage:
  python filter_best_per_rank.py \
      --summary /path/to/OUT_DIR/FINAL_gb_summary.txt \
      --out /path/to/OUT_DIR/FILTERED_best_per_rank.txt
"""

import csv
import argparse
from collections import defaultdict


def parse_args():
    ap = argparse.ArgumentParser(
        description="Pick the best group per slab rank where BOTH grains have ac-contact planes."
    )
    ap.add_argument("--summary", required=True, help="Path to FINAL_gb_summary.txt")
    ap.add_argument("--out", required=True, help="Path to write FILTERED_best_per_rank.txt")
    return ap.parse_args()


# ---------- contact-plane parsing helpers ----------

def _normalize_plane_label(label: str) -> str:
    """
    Normalize a single plane label like 'ac', 'AC', ' ca ' etc.
    Returns lowercased, stripped string.
    """
    return label.strip().lower()


def _extract_planes_g1_g2(field: str):
    """
    Parse a contact_plane field that may look like:
        'g1=ac,g2=ac'
        'g1=ab,g2=ac'
        'g1=ac , g2=ca'
        'ac,ac'
        'ac,ab'
    Returns a list of plane labels, e.g. ['ac', 'ac'].
    If parsing fails or there are no valid labels, returns [].
    """
    if not field:
        return []

    lab = field.strip().lower()

    # Treat "na" or similar as no data
    if lab in ("na", "none", "null", "-"):
        return []

    # Remove some common separators/spaces to simplify
    # but keep comma because we use it to split.
    for sep in [" ", ";"]:
        lab = lab.replace(sep, "")

    # Now split on commas: 'g1=ac,g2=ac' -> ['g1=ac', 'g2=ac']
    parts = [p for p in lab.split(",") if p]

    planes = []
    for p in parts:
        # 'g1=ac' -> 'ac'
        if "=" in p:
            _, val = p.split("=", 1)
            planes.append(_normalize_plane_label(val))
        else:
            # Plain label: 'ac'
            planes.append(_normalize_plane_label(p))

    # Only keep non-empty labels
    return [pl for pl in planes if pl]


def is_both_ac(field: str) -> bool:
    """
    Return True iff BOTH grains are 'ac'-type planes.

    For new-style labels:
        'g1=ac,g2=ac'  -> True
        'g1=ac,g2=ca'  -> True (treat 'ca' same as 'ac' family)
        'g1=ab,g2=ac'  -> False
        'g1=ac,g2=ab'  -> False

    For old-style labels (if they ever occur):
        'ac' or 'ca'   -> treated as a single 'ac'-type plane, but
                          DOES NOT guarantee both grains; we can choose
                          to reject or accept. Here we accept only if
                          we see exactly 2 planes and both are ac/ca.
    """
    planes = _extract_planes_g1_g2(field)

    # Require exactly two planes (g1 and g2); if not, we don't know
    # both sides -> reject to be strict.
    if len(planes) != 2:
        return False

    # Accept if BOTH are 'ac'-family (ac or ca)
    return all(pl in ("ac", "ca") for pl in planes)


# ---------- ranking / tie-breakers ----------

def row_key_for_tiebreakers(row):
    """
    Sort key: (dist_to_center ASC, min(G1,G2) DESC, (G1+G2) DESC, GB_N DESC)
    """
    dist = float(row["dist_to_box_center_A"])
    g1 = int(row["G1_N"])
    g2 = int(row["G2_N"])
    gb = int(row["GB_N"])
    return (dist, -min(g1, g2), -(g1 + g2), -gb)


# ---------- main ----------

def main():
    args = parse_args()

    # Read the summary file
    with open(args.summary, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [r for r in reader]

    if not rows:
        raise ValueError(f"No data rows found in {args.summary}")

    # Normalize/validate essential fields
    needed = {
        "slab_rank", "slab_dir", "slab_stem", "GB_AXIS", "y_center_A",
        "dist_to_box_center_A", "GB_N", "G1_N", "G2_N", "contact_plane", "misori_deg"
    }
    missing = needed - set(rows[0].keys())
    if missing:
        raise ValueError(f"Input file missing required columns: {sorted(missing)}")

    # Filter by "both grains ac" condition
    ac_rows = [r for r in rows if is_both_ac(r.get("contact_plane", ""))]

    # Group by slab_rank
    by_rank = defaultdict(list)
    for r in ac_rows:
        try:
            r["slab_rank"] = int(r["slab_rank"])
            # Defensive numeric parsing
            r["y_center_A"] = float(r["y_center_A"])
            r["dist_to_box_center_A"] = float(r["dist_to_box_center_A"])
            r["GB_N"] = int(r["GB_N"])
            r["G1_N"] = int(r["G1_N"])
            r["G2_N"] = int(r["G2_N"])
        except Exception:
            # Skip malformed lines quietly
            continue
        by_rank[r["slab_rank"]].append(r)

    # Choose best per rank
    best_rows = []
    for rank, rlist in by_rank.items():
        if not rlist:
            continue
        rlist.sort(key=row_key_for_tiebreakers)
        best_rows.append(rlist[0])

    # Sort output by slab_rank ascending
    best_rows.sort(key=lambda r: r["slab_rank"])

    # Write filtered TSV
    out_cols = [
        "slab_rank", "slab_dir", "slab_stem", "GB_AXIS",
        "y_center_A", "dist_to_box_center_A", "GB_N", "G1_N", "G2_N", "contact_plane", "misori_deg"
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=out_cols)
        w.writeheader()
        for r in best_rows:
            w.writerow({c: r[c] for c in out_cols})

    # Console summary
    print(f"[done] Wrote best-per-rank selection → {args.out}")
    print(f"Ranks covered (with g1=ac,g2=ac): {len(best_rows)}")
    if best_rows:
        closest = min(best_rows, key=lambda r: r["dist_to_box_center_A"])
        print(
            f"Closest to box center: rank {closest['slab_rank']} "
            f"({closest['slab_dir']}, {closest['slab_stem']}) "
            f"Δ={closest['dist_to_box_center_A']:.2f} Å | "
            f"G1={closest['G1_N']} G2={closest['G2_N']} GB={closest['GB_N']} | "
            f"planes={closest['contact_plane']}"
        )
    else:
        print(
            "No rows matched condition g1=ac,g2=ac (or ca). "
            "Check your FINAL_gb_summary.txt or contact_plane labels."
        )


if __name__ == "__main__":
    main()
