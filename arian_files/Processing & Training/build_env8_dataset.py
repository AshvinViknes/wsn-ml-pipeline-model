# build_env8_dataset.py
# Scenario-1 (Env) dataset with 8 channels per window:
# [dRSSI, dLQI, RSSI, LQI, MA_short_RSSI, MA_short_LQI, MA_long_RSSI, MA_long_LQI]

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Build 8-ch env dataset from cleaned CSVs")
    ap.add_argument("--cleaned", default="cleaned", help="Folder with *_cleaned.csv")
    ap.add_argument("--outdir",  default="processed_v2_10s_8ch", help="Output folder")
    # Defaults for 10 s windows, 50% overlap, 100 samples (≈10 Hz grid)
    ap.add_argument("--win_ms", type=int, default=10_000, help="Window length in ms")
    ap.add_argument("--step_ms", type=int, default=5_000,  help="Hop length in ms")
    ap.add_argument("--n_samples", type=int, default=100,  help="Samples per window")
    # Smoothing windows in seconds (moving averages)
    ap.add_argument("--ma_short_s", type=float, default=1.5)
    ap.add_argument("--ma_long_s",  type=float, default=5.0)
    return ap.parse_args()

NAME_RE = re.compile(r"^([ABC])_([ABC])_([1-5])(?:_cleaned)?$", re.I)

def resample_window(t_ms, v, t0, t1, n):
    """Interpolate v(t) onto n evenly spaced points in [t0, t1]."""
    mask = (t_ms >= t0) & (t_ms <= t1)
    if mask.sum() < 2:
        return None
    x = t_ms[mask].astype(np.float64)
    y = v[mask].astype(np.float64)
    xu, idx = np.unique(x, return_index=True)
    yu = y[idx]
    if xu.size < 2:
        return None
    grid = np.linspace(float(t0), float(t1), int(n), dtype=np.float64)
    out = np.interp(grid, xu, yu)
    return out.astype(np.float32), grid  # keep grid float64 for numerics

def moving_average_on_grid(vals, grid_ms, win_s):
    """Centered boxcar MA on a uniform grid; robust to edge/degenerate cases."""
    if win_s <= 0: return vals
    T = len(vals)
    if T < 3: return vals
    span = float(grid_ms[-1] - grid_ms[0])
    if not np.isfinite(span) or span <= 0: return vals
    dt = span / max(1, T - 1)
    if not np.isfinite(dt) or dt <= 0: return vals
    r = int(round((win_s * 1000.0 / dt) / 2.0))
    r = max(1, min(r, (T - 1) // 2))
    k = 2 * r + 1
    kernel = np.ones(k, dtype=np.float32) / k
    out = np.convolve(vals, kernel, mode="same").astype(np.float32)
    return out

def frames_from_df(df, win_ms, step_ms, n_samples, ma_short_s, ma_long_s):
    """Yield (8, n_samples) frames per window."""
    t = df["timestamp_ms"].to_numpy()
    rssi = df["rssi"].to_numpy()
    lqi  = df["lqi"].to_numpy()
    if len(t) < 2: return []

    frames = []
    cur = t.min()
    tN  = t.max()
    while cur + win_ms <= tN:
        rssi_u = resample_window(t, rssi, cur, cur + win_ms, n_samples)
        lqi_u  = resample_window(t, lqi,  cur, cur + win_ms, n_samples)
        if rssi_u is not None and lqi_u is not None:
            rssi_vals, grid = rssi_u
            lqi_vals,  _    = lqi_u
            drssi = np.diff(rssi_vals, prepend=rssi_vals[0])
            dlqi  = np.diff(lqi_vals,  prepend=lqi_vals[0])

            maS_rssi = moving_average_on_grid(rssi_vals, grid, ma_short_s)
            maS_lqi  = moving_average_on_grid(lqi_vals,  grid, ma_short_s)
            maL_rssi = moving_average_on_grid(rssi_vals, grid, ma_long_s)
            maL_lqi  = moving_average_on_grid(lqi_vals,  grid, ma_long_s)

            frame = np.stack(
                [drssi, dlqi, rssi_vals, lqi_vals, maS_rssi, maS_lqi, maL_rssi, maL_lqi],
                axis=0
            ).astype(np.float32)
            frames.append(frame)
        cur += step_ms
    return frames

def main():
    args = parse_args()
    in_dir  = Path(args.cleaned)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_list, y_list, meta_rows = [], [], []
    csvs = sorted(in_dir.glob("*.csv"))
    if not csvs:
        print(f"[WARN] No CSV files found in {in_dir.resolve()}")

    for csv in csvs:
        stem = csv.stem.replace("_cleaned", "")
        m = NAME_RE.match(stem)
        if not m:
            print(f"[Skip] {csv.name} (name not recognized)"); continue
        TX, RX, Env = m.group(1).upper(), m.group(2).upper(), int(m.group(3))

        try:
            df = pd.read_csv(csv)
        except Exception as e:
            print(f"[ERR] {csv.name}: {e}"); continue

        need = {"timestamp_ms","rssi","lqi"}
        if not need.issubset(df.columns):
            print(f"[ERR] Missing columns in {csv.name} -> {need}"); continue

        df = df.dropna(subset=list(need)).sort_values("timestamp_ms")
        if df.empty:
            print(f"[Skip] Empty after cleaning: {csv.name}"); continue

        frames = frames_from_df(df, args.win_ms, args.step_ms, args.n_samples,
                                args.ma_short_s, args.ma_long_s)
        if not frames:
            print(f"[Skip] No frames generated: {csv.name}"); continue

        Xf = np.stack(frames, axis=0)  # (n_frames, 8, n_samples)
        X_list.append(Xf)
        y_list.append(np.full(Xf.shape[0], Env, np.int64))
        meta_rows.extend([{"file": stem, "TX": TX, "RX": RX, "Env": Env}] * Xf.shape[0])
        print(f"[OK] {csv.name}: {Xf.shape[0]} frames")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    meta = pd.DataFrame(meta_rows)

    np.save(out_dir / "X_env8_unscaled.npy", X)
    np.save(out_dir / "y_env.npy", y)
    meta.to_csv(out_dir / "meta_env.csv", index=False)

    print("Saved:")
    print("  ", (out_dir / "X_env8_unscaled.npy").resolve(), X.shape)
    print("  ", (out_dir / "y_env.npy").resolve(), y.shape)
    print("  ", (out_dir / "meta_env.csv").resolve(), meta.shape)

if __name__ == "__main__":
    main()
