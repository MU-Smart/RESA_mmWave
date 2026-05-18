"""
qc_labeled_radar.py

Quality-control diagnostics for labeled_radar_points_v3.csv.

Prints a concise report covering:
  - Total point count
  - Bucket (navigation class) distribution with percentages
  - Top ADE20K label distribution
  - Majority-fraction statistics (label boundary quality)
  - Camera-depth (cam_z) statistics (useful for spotting bad projections)
  - Doppler statistics per bucket (if doppler column present)
  - SNR statistics per bucket (if snr column present)
  - Per-session breakdown (if multiple sessions were merged)
  - Frames-per-bucket: how many unique video frames contributed each bucket

Usage:
    python qc_labeled_radar.py
        --labeled_csv "C:\\...\\Processing\\session_...\\labeled_radar_points_v3.csv"
        [--top_ade 20]
        [--plot]   # future: histogram plots (requires matplotlib)

Dependencies:
    pip install pandas numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pct(n: int, total: int) -> str:
    if total == 0:
        return "  n/a"
    return f"{100 * n / total:5.1f}%"


def print_separator(char: str = "-", width: int = 60) -> None:
    print(char * width)


def main():
    ap = argparse.ArgumentParser(
        description="Quality-control report for labeled radar points CSV."
    )
    ap.add_argument("--labeled_csv", required=True,
                    help="Path to labeled_radar_points_v3.csv")
    ap.add_argument("--top_ade", type=int, default=20,
                    help="Number of top ADE20K labels to show. Default: 20")
    args = ap.parse_args()

    csv_path = Path(args.labeled_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    print("=" * 65)
    print(f"QC Report: {csv_path.name}")
    print("=" * 65)

    # ---- Basic ----
    N = len(df)
    print(f"\nTotal labeled points : {N:,}")
    print(f"CSV path             : {csv_path}")

    cols = list(df.columns)
    print(f"Columns              : {cols}")

    # ---- Bucket distribution ----
    print_separator()
    print("BUCKET DISTRIBUTION")
    print_separator()
    bc = df["bucket"].value_counts(dropna=False)
    for bucket, count in bc.items():
        bar = "█" * min(40, int(40 * count / N))
        print(f"  {str(bucket):<12} {count:7,}  {pct(count, N)}  {bar}")

    # ---- Top ADE labels ----
    print_separator()
    print(f"TOP {args.top_ade} ADE20K LABELS")
    print_separator()
    if "ade_name" in df.columns:
        ac = df["ade_name"].value_counts(dropna=False).head(args.top_ade)
        for label, count in ac.items():
            print(f"  {str(label):<35} {count:7,}  {pct(count, N)}")
    else:
        print("  (ade_name column not found)")

    # ---- Majority fraction quality ----
    print_separator()
    print("LABEL BOUNDARY QUALITY  (maj_frac)")
    print_separator()
    if "maj_frac" in df.columns:
        mf = df["maj_frac"].describe()
        for stat in ("min", "25%", "50%", "75%", "max", "mean"):
            print(f"  {stat:<8}: {mf[stat]:.4f}")
        low = (df["maj_frac"] < 0.60).sum()
        print(f"\n  Points with maj_frac < 0.60 : {low:,}  "
              f"({pct(low, N)}) — these were already filtered if min_majority=0.60")
    else:
        print("  (maj_frac column not found)")

    # ---- Camera depth ----
    print_separator()
    print("CAMERA DEPTH  (cam_z, metres)")
    print_separator()
    if "cam_z" in df.columns:
        cz = df["cam_z"].describe()
        for stat in ("min", "25%", "50%", "75%", "max", "mean"):
            print(f"  {stat:<8}: {cz[stat]:.3f} m")
        neg = (df["cam_z"] <= 0).sum()
        if neg > 0:
            print(f"\n  [WARN] {neg} points with cam_z <= 0 — behind-camera projections "
                  f"(should have been filtered).")
        far = (df["cam_z"] > 8.0).sum()
        if far > 0:
            print(f"  [INFO] {far} points at cam_z > 8.0 m — may be noisy at long range.")
    else:
        print("  (cam_z column not found)")

    # ---- Doppler per bucket ----
    if "doppler" in df.columns:
        print_separator()
        print("DOPPLER (m/s) PER BUCKET  (median | mean)")
        print_separator()
        for bucket, grp in df.groupby("bucket"):
            d = grp["doppler"].dropna()
            if len(d) == 0:
                continue
            moving = (d.abs() > 0.05).sum()
            print(f"  {str(bucket):<12}  n={len(d):6,}  "
                  f"median={d.median():+.3f}  mean={d.mean():+.3f}  "
                  f"|dop|>0.05: {pct(moving, len(d))}")

    # ---- SNR per bucket ----
    if "snr" in df.columns:
        print_separator()
        print("SNR (dB) PER BUCKET  (median | mean)")
        print_separator()
        for bucket, grp in df.groupby("bucket"):
            s = grp["snr"].dropna()
            if len(s) == 0:
                continue
            print(f"  {str(bucket):<12}  n={len(s):6,}  "
                  f"median={s.median():5.1f}  mean={s.mean():5.1f}  "
                  f"min={s.min():4.1f}  max={s.max():5.1f}")

    # ---- Per-session breakdown ----
    if "session" in df.columns:
        sessions = df["session"].unique()
        if len(sessions) > 1:
            print_separator()
            print(f"PER-SESSION BREAKDOWN  ({len(sessions)} sessions)")
            print_separator()
            for sess in sorted(sessions):
                sub = df[df["session"] == sess]
                bc_s = sub["bucket"].value_counts()
                summary = "  ".join(f"{b}:{c}" for b, c in bc_s.items())
                print(f"  {str(sess)[-30:]:>30}  n={len(sub):6,}  {summary}")

    # ---- Unique frames per bucket ----
    print_separator()
    print("UNIQUE VIDEO FRAMES PER BUCKET")
    print_separator()
    if "video_frame_index" in df.columns:
        for bucket, grp in df.groupby("bucket"):
            n_frames = grp["video_frame_index"].nunique()
            n_pts    = len(grp)
            print(f"  {str(bucket):<12}  {n_frames:5,} frames  "
                  f"avg {n_pts / max(n_frames, 1):.1f} pts/frame")

    # ---- XYZ range sanity ----
    print_separator()
    print("RADAR XYZ RANGE SANITY")
    print_separator()
    for ax in ("x", "y", "z"):
        if ax in df.columns:
            col = df[ax]
            print(f"  {ax}: min={col.min():.2f}  max={col.max():.2f}  "
                  f"mean={col.mean():.2f}  std={col.std():.2f}")

    print("=" * 65)
    print("QC done.")

    # ---- Actionable recommendations ----
    issues = []
    if "bucket" in df.columns:
        bc2 = df["bucket"].value_counts()
        dominant = bc2.index[0] if len(bc2) > 0 else None
        if dominant and bc2[dominant] / N > 0.80:
            issues.append(
                f"[IMBALANCE] '{dominant}' is {pct(bc2[dominant], N)} of all points. "
                f"Apply weighted sampling or loss weighting in training."
            )
        if bc2.get("human", 0) == 0:
            issues.append(
                "[MISSING CLASS] No 'human' points found. "
                "Include sessions with people walking in the scene."
            )
        if bc2.get("door", 0) < 50:
            issues.append(
                f"[LOW CLASS] 'door' has only {bc2.get('door', 0)} points. "
                "Record more sessions in front of doorways."
            )

    if issues:
        print("\n[RECOMMENDATIONS]")
        for issue in issues:
            print(f"  {issue}")


if __name__ == "__main__":
    main()
