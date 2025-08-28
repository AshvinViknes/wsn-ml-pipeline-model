# preprocessing_workflow.py
# Parse pyterm .txt logs (e.g., "2025-07-11 16:20:48,864 # -56,148") into clean CSVs
# with columns: timestamp_ms, rssi, lqi. Also accepts already-clean CSVs.
import argparse, re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

NAME_RE = re.compile(r"^([ABC])_([ABC])_([1-5])(?:\.(?:txt|csv))?$", re.I)
LINE_RE = re.compile(
    r"^\s*(\d{4}-\d{2}-\d{2})\s+"
    r"(\d{2}:\d{2}:\d{2}),(\d{3})\s*#\s*"
    r"(-?\d+)\s*,\s*(-?\d+)\s*$"
)

def parse_args():
    ap = argparse.ArgumentParser(description="Convert pyterm .txt logs -> cleaned CSVs")
    ap.add_argument("--in", dest="inp", required=True, help="Folder with A_B_1.txt etc.")
    ap.add_argument("--out", dest="out", required=True, help="Output folder for *_cleaned.csv")
    ap.add_argument("--patterns", default="*.txt,*.csv",
                    help="Comma-separated glob patterns to read (default: *.txt,*.csv)")
    return ap.parse_args()

def parse_pyterm_txt(path: Path) -> pd.DataFrame:
    ts = []; rssi = []; lqi = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue  # skip banner, connect lines, or noise
            d, t, ms, rs, lq = m.groups()
            # build wallclock datetime then convert to milliseconds relative to first sample
            dt = datetime.strptime(f"{d} {t}.{ms}", "%Y-%m-%d %H:%M:%S.%f")
            ts.append(dt); rssi.append(int(rs)); lqi.append(int(lq))
    if not ts:
        raise ValueError(f"No valid samples parsed from {path.name}")
    ts0 = ts[0]
    ts_ms = np.array([(t - ts0).total_seconds()*1000.0 for t in ts], dtype=np.float64)
    # de-duplicate by timestamp if needed (keep first)
    df = pd.DataFrame({"timestamp_ms": ts_ms, "rssi": rssi, "lqi": lqi})
    df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms").reset_index(drop=True)
    # ensure numeric dtypes
    df["timestamp_ms"] = df["timestamp_ms"].astype(np.int64)
    df["rssi"] = df["rssi"].astype(np.int16)
    df["lqi"] = df["lqi"].astype(np.int16)
    return df

def load_csv_if_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # map common header variants
    ts_col = cols.get("timestamp_ms") or cols.get("ts_ms") or cols.get("timestamp")
    rssi_col = cols.get("rssi")
    lqi_col  = cols.get("lqi")
    if not (ts_col and rssi_col and lqi_col):
        raise ValueError(f"CSV missing required columns in {path.name}")
    out = df[[ts_col, rssi_col, lqi_col]].copy()
    out.columns = ["timestamp_ms", "rssi", "lqi"]
    out["timestamp_ms"] = out["timestamp_ms"].astype(np.int64)
    out["rssi"] = out["rssi"].astype(np.int16)
    out["lqi"] = out["lqi"].astype(np.int16)
    out = out.dropna().sort_values("timestamp_ms").reset_index(drop=True)
    return out

def main():
    args = parse_args()
    in_dir  = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]

    files = []
    for pat in patterns:
        files.extend(sorted(in_dir.glob(pat)))

    if not files:
        print(f"[WARN] No files matched in {in_dir.resolve()} with {patterns}")
        print("Done. Cleaned rows total: 0"); return

    total_rows = 0
    for src in files:
        stem = src.name
        base = src.stem  # e.g., A_B_1
        m = NAME_RE.match(stem)
        if not m:
            print(f"[Skip] {stem} (name must look like A_B_1.txt)"); continue

        try:
            if src.suffix.lower() == ".txt":
                df = parse_pyterm_txt(src)
            else:
                df = load_csv_if_clean(src)
        except Exception as e:
            print(f"[ERR] {stem}: {e}")
            continue

        dst = out_dir / f"{base}_cleaned.csv"
        df.to_csv(dst, index=False)
        total_rows += len(df)
        print(f"[OK] {stem} -> {dst.name} ({len(df)} rows)")

    print(f"Done. Cleaned rows total: {total_rows}")

if __name__ == "__main__":
    main()
