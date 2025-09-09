#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLEANING & WINDOWING for WSN logs
- Clean: TXT -> CSV with columns [timestamp_ms, rssi, lqi]
- Window: CSV -> overlapping windows, linearly resampled to a fixed grid
Only relies on the actual data lines that look like:
    YYYY-MM-DD HH:MM:SS,fff # -72,84
"""

import argparse
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd


# --------- Parsing: regex just for real data lines ----------
LINE_RE = re.compile(
    r'^(?P<dt>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,\.]\d+)'
    r'\s+#\s+(?P<rssi>-?\d+)[,;\s]+(?P<lqi>\d+)\s*$'
)

def parse_txt_file(txt_path: Path, debug=False) -> pd.DataFrame | None:
    """Return tidy dataframe [timestamp_ms, rssi, lqi] or None if no data."""
    ts, rssi, lqi = [], [], []
    with txt_path.open("r", errors="replace") as fh:
        for line in fh:
            m = LINE_RE.match(line)
            if not m:
                continue
            dt_str = m.group("dt").replace(".", ",")  # support both ',' and '.'
            try:
                # Treat as naive local time; consistent within each file.
                t = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S,%f")
                t_ms = int(t.timestamp() * 1000)
            except Exception:
                continue
            ts.append(t_ms)
            rssi.append(int(m.group("rssi")))
            lqi.append(int(m.group("lqi")))

    if not ts:
        if debug:
            print(f"[DBG] {txt_path.name}: no data lines matched")
        return None

    df = pd.DataFrame({"timestamp_ms": ts, "rssi": rssi, "lqi": lqi})
    # basic sanitization
    df = (
        df.dropna()
          .drop_duplicates()
          .sort_values("timestamp_ms")
    )
    # very light sanity range checks
    df = df[(df["rssi"].between(-120, 0)) & (df["lqi"].between(0, 255))]
    if len(df) < 3:
        if debug:
            print(f"[DBG] {txt_path.name}: <3 rows after sanitization")
        return None
    return df


# --------- Windowing ----------
def make_windows(df: pd.DataFrame, win_ms: int, step_ms: int, n_samples: int):
    """Yield (start_ms, ndarray[n_samples, 2]) for (rssi, lqi) per window."""
    t = df["timestamp_ms"].to_numpy(dtype=np.int64)
    r = df["rssi"].to_numpy(dtype=float)
    q = df["lqi"].to_numpy(dtype=float)
    if len(t) < 2:
        return

    t0_all, tN = t[0], t[-1]
    # slide [start, start+win)
    start = t0_all
    while start + win_ms <= tN:
        left = np.searchsorted(t, start, side="left")
        right = np.searchsorted(t, start + win_ms, side="left")
        if right - left >= 2:
            t_seg = t[left:right].astype(float)
            r_seg = r[left:right]
            q_seg = q[left:right]
            grid = np.linspace(start, start + win_ms, n_samples, endpoint=False)
            r_g = np.interp(grid, t_seg, r_seg)
            q_g = np.interp(grid, t_seg, q_seg)
            X = np.stack([r_g, q_g], axis=1)  # [n_samples, 2] -> (rssi, lqi)
            yield (start, X)
        start += step_ms


# --------- CLI workflow ----------
def main():
    ap = argparse.ArgumentParser(
        description="Clean WSN TXT logs and produce fixed-shape windows."
    )
    ap.add_argument("--raw", required=True, help="Folder with .txt files")
    ap.add_argument("--cleaned", required=True, help="Output folder for cleaned CSVs")
    ap.add_argument("--out", required=True, help="Output folder for windows (.npy) + manifest.csv")
    ap.add_argument("--win_ms", type=int, default=10_000, help="Window length in ms (default 10000)")
    ap.add_argument("--step_ms", type=int, default=5_000, help="Hop size in ms (default 5000)")
    ap.add_argument("--n_samples", type=int, default=100, help="Samples per window (default 100)")
    ap.add_argument("--debug", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    cleaned_dir = Path(args.cleaned)
    out_dir = Path(args.out)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) CLEANING
    txt_files = sorted(raw_dir.rglob("*.txt"))
    if args.debug:
        print(f"[DBG] discovered {len(txt_files)} TXT files (recursive) under {raw_dir.name}")

    ok_clean, bad = 0, 0
    for p in txt_files:
        df = parse_txt_file(p, debug=args.debug)
        if df is None:
            bad += 1
            if args.debug:
                print(f"[DBG] {p.name}: schema/cleaning failed")
            continue
        out_csv = cleaned_dir / f"{p.stem}_cleaned.csv"
        df.to_csv(out_csv, index=False)
        ok_clean += 1
        if args.debug:
            span = (df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]) / 1000
            print(f"[DBG] cleaned {p.name} -> {out_csv.name} | rows={len(df)} | span≈{span:.1f}s")
    print(f"Cleaning done. Files: {len(txt_files)} | OK: {ok_clean} | Bad/empty: {bad}")

    # 2) WINDOWING
    cleaned_csvs = sorted(cleaned_dir.glob("*_cleaned.csv"))
    if not cleaned_csvs:
        print(f"[WARN] No cleaned CSVs found in {cleaned_dir}")
        return

    manifest_rows = []
    for csv in cleaned_csvs:
        df = pd.read_csv(csv)
        wins = list(make_windows(df, args.win_ms, args.step_ms, args.n_samples))
        if args.debug:
            print(f"[DBG] {csv.name}: produced {len(wins)} windows")
        base = csv.stem.replace("_cleaned", "")
        # basic tags from file name, if available
        m = re.match(r"([A-Za-z])_([A-Za-z])_(\d+)", base)
        tx, rx, env = (m.group(1), m.group(2), m.group(3)) if m else ("?", "?", "?")

        for start_ms, X in wins:
            npy_path = out_dir / f"{base}__t{start_ms}.npy"
            np.save(npy_path, X)
            manifest_rows.append(
                {
                    "win_path": str(npy_path),
                    "file_id": base,
                    "start_ms": int(start_ms),
                    "end_ms": int(start_ms + args.win_ms),
                    "tx": tx,
                    "rx": rx,
                    "env": env,
                }
            )

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
        print(f"Windowing done. Windows: {len(manifest_rows)} | Saved: {out_dir/'manifest.csv'}")
    else:
        print("[WARN] No windows were produced. Check cleaning output and window params.")


if __name__ == "__main__":
    main()
