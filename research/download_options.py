from __future__ import annotations

import argparse
import datetime as dt
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/option/daily/EOHSummary/BTCUSDT"


def dates(start: str, end: str):
    d = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    while d <= last:
        yield d
        d += dt.timedelta(days=1)


def download(start: str, end: str, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for d in dates(start, end):
        name = f"BTCUSDT-EOHSummary-{d:%Y-%m-%d}.zip"
        target = out / name
        if target.exists():
            continue
        url = f"{BASE}/{name}"
        r = requests.get(url, timeout=60)
        if r.status_code == 404:
            print(f"missing {d}")
            continue
        r.raise_for_status()
        target.write_bytes(r.content)
        print(f"downloaded {name} ({len(r.content):,} bytes)")


def combine(raw: Path, output: Path):
    frames = []
    for zpath in sorted(raw.glob("*.zip")):
        with zipfile.ZipFile(zpath) as z:
            for member in z.namelist():
                if member.lower().endswith((".csv", ".csv.gz")):
                    with z.open(member) as f:
                        frames.append(pd.read_csv(f))
    if not frames:
        raise RuntimeError("No option CSV files found")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates()
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"combined {len(df):,} rows -> {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--raw", default="data/raw/options")
    p.add_argument("--output", default="data/processed/options.parquet")
    p.add_argument("--combine", action="store_true")
    a = p.parse_args()
    download(a.start, a.end, Path(a.raw))
    if a.combine:
        combine(Path(a.raw), Path(a.output))
