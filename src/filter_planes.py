#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter the final GB summary to one best group per slab rank.

Selection criteria (applied in order):
  0) KEEP ONLY rows where the contact_plane matches a user-specified pair,
     e.g. "ac-ac" meaning g1=ac, g2=ac (with 'ca' treated as 'ac'-family).
  1) minimal dist_to_box_center_A
  2) maximal min(G1,G2)         (balance: both grains large)
  3) maximal (G1_N + G2_N)      (total grain size)
  4) maximal GB_N               (size of GB set)

This refactor preserves exact behavior but uses structured logging and
adds small docstrings and type hints.
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------- contact-plane parsing helpers ----------


def _normalize_plane_label(label: str) -> str:
    """
    Normalize a single plane label like 'ac', 'AC', ' ca ' etc.
    Returns lowercased, stripped string.
    """
    return label.strip().lower()


def _extract_planes_g1_g2(field: str) -> List[str]:
    """
    Parse a contact_plane field into a list of plane labels (g1, g2).
    Returns list of labels (may be shorter than 2 if parsing fails).
    """
    if not field:
        return []

    lab = field.strip().lower()

    # Treat "na" or similar as no data
    if lab in ("na", "none", "null", "-", "nan"):
        return []

    # Remove spaces and semicolons, keep commas
    for sep in [" ", ";"]:
        lab = lab.replace(sep, "")

    # Split on commas: 'g1=ac,g2=ac' -> ['g1=ac', 'g2=ac']
    parts = [p for p in lab.split(",") if p]

    planes: List[str] = []
    for p in parts:
        # 'g1=ac' -> 'ac'
        if "=" in p:
            _, val = p.split("=", 1)
            planes.append(_normalize_plane_label(val))
        else:
            # Plain label: 'ac'
            planes.append(_normalize_plane_label(p))

    return [pl for pl in planes if pl]


def _parse_plane_pair_spec(spec: str) -> Tuple[str, str]:
    """
    Parse a user plane-pair spec like 'ac-ac' into (plane1, plane2).
    '*' is allowed as a wildcard.
    """
    if not spec:
        return "*", "*"

    s = spec.strip().lower()

    # If it's already in 'g1=ac,g2=ac' style, reuse the same parser
    if "g1" in s or "g2" in s:
        planes = _extract_planes_g1_g2(s)
        if len(planes) == 1:
            return planes[0], planes[0]
        if len(planes) >= 2:
            return planes[0], planes[1]

    # Normalize separators to '-'
    for sep in [",", "/", ";"]:
        s = s.replace(sep, "-")

    parts = [p for p in s.split("-") if p]
    if len(parts) == 1:
        return parts[0], parts[0]
    # if more than 2, just take the first two
    return parts[0], parts[1]


def _match_single_plane(actual: str, pattern: str) -> bool:
    """
    Compare one actual plane with one pattern token.

    Rules:
      - pattern '*'  → matches anything
      - 'ac' and 'ca' are treated as equivalent family
      - otherwise: exact match
    """
    actual = _normalize_plane_label(actual)
    pattern = _normalize_plane_label(pattern)

    if pattern == "*":
        return True

    # treat ac/ca as equivalent
    if pattern in ("ac", "ca"):
        return actual in ("ac", "ca")

    return actual == pattern


def match_plane_pair(field: str, plane_pair: str) -> bool:
    """
    Return True if contact_plane field matches the user pattern.

    plane_pair examples: 'ac-ac', 'ab-ac', 'ac-*'
    """
    planes_actual = _extract_planes_g1_g2(field)
    if len(planes_actual) != 2:
        # We require exactly 2 planes (g1, g2). If not, reject strictly.
        return False

    p1, p2 = _parse_plane_pair_spec(plane_pair)
    return _match_single_plane(planes_actual[0], p1) and _match_single_plane(planes_actual[1], p2)


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


# ---------- main API ----------


def filter_best_per_rank(
    summary_path: Path | str,
    out_path: Path | str,
    plane_pair: str = "ac-ac",
) -> None:
    """
    Filter FINAL_gb_summary.txt to one best row per slab_rank matching
    a given contact_plane pattern, and write a filtered TSV.

    Parameters
    ----------
    summary_path : Path or str
        Path to FINAL_gb_summary.txt written by main pipeline.
    out_path : Path or str
        Output TSV path.
    plane_pair : str
        Desired plane pattern, e.g. "ac-ac", "ab-ac", "ac-*".
    """
    summary_path = Path(summary_path)
    out_path = Path(out_path)

    # Read the summary file
    with summary_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [r for r in reader]

    if not rows:
        raise ValueError(f"No data rows found in {summary_path}")

    # Validate essential fields
    needed = {
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
    }
    missing = needed - set(rows[0].keys())
    if missing:
        raise ValueError(f"Input file missing required columns: {sorted(missing)}")

    # Filter by contact_plane pattern
    matched_rows = [r for r in rows if match_plane_pair(r.get("contact_plane", ""), plane_pair)]

    # Group by slab_rank
    by_rank: dict[int, list] = defaultdict(list)
    for r in matched_rows:
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
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=out_cols)
        w.writeheader()
        for r in best_rows:
            w.writerow({c: r[c] for c in out_cols})

    logger.info("Wrote best-per-rank selection → %s", out_path)
    logger.info("Ranks covered (matching plane_pair='%s'): %d", plane_pair, len(best_rows))
    if best_rows:
        closest = min(best_rows, key=lambda r: r["dist_to_box_center_A"])
        logger.info(
            "Closest to box center: slab %s (%s, %s) Δ=%.2f Å | G1=%s G2=%s GB=%s | planes=%s",
            closest["slab_rank"],
            closest["slab_dir"],
            closest["slab_stem"],
            closest["dist_to_box_center_A"],
            closest["G1_N"],
            closest["G2_N"],
            closest["GB_N"],
            closest["contact_plane"],
        )
    else:
        logger.info(
            "No rows matched the requested plane_pair ('%s'). Check your FINAL_gb_summary.txt or labels.",
            plane_pair,
        )