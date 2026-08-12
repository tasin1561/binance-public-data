from __future__ import annotations

import re
import sys
import pandas as pd

SYMBOL_RE = re.compile(r"^BTC-(\d{6})-([0-9.]+)-(C|P)$", re.IGNORECASE)


def main(path: str) -> None:
    df = pd.read_parquet(path)
    print("rows:", len(df))
    print("columns:", ", ".join(map(str, df.columns)))
    if df.empty:
        raise SystemExit("Dataset is empty")

    cols = {str(c).lower() for c in df.columns}
    required = {"symbol", "underlying", "mark_price", "strike", "type"}
    missing = required - cols
    if missing:
        raise SystemExit(f"Missing required Binance EOH fields: {sorted(missing)}")

    if "date" in cols and "hour" in cols:
        date_col = next(c for c in df.columns if str(c).lower() == "date")
        hour_col = next(c for c in df.columns if str(c).lower() == "hour")
        t = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.normalize() + pd.to_timedelta(pd.to_numeric(df[hour_col], errors="coerce"), unit="h")
    else:
        time_col = next((c for c in df.columns if str(c).lower() in {"timestamp", "time", "open_time"}), None)
        if time_col is None:
            raise SystemExit("No timestamp or date+hour fields found")
        t = pd.to_datetime(pd.to_numeric(df[time_col], errors="coerce"), unit="ms", errors="coerce", utc=True)

    symbol_col = next(c for c in df.columns if str(c).lower() == "symbol")
    type_col = next(c for c in df.columns if str(c).lower() == "type")
    symbols = df[symbol_col].astype(str)
    parsed = symbols.str.extract(SYMBOL_RE)
    types = df[type_col].astype(str).str.upper().str.strip()
    valid_types = types.isin({"CALL", "C", "PUT", "P"})

    print("time_min:", t.min())
    print("time_max:", t.max())
    print("unique_symbols:", df[symbol_col].nunique())
    print("calls:", int(types.isin({"CALL", "C"}).sum()))
    print("puts:", int(types.isin({"PUT", "P"}).sum()))
    print("parseable_symbol_expiries:", int(parsed[0].notna().sum()))
    print("invalid_option_types:", int((~valid_types).sum()))
    print("invalid_timestamps:", int(t.isna().sum()))

    if parsed[0].isna().any():
        raise SystemExit("Some Binance option symbols have unparseable expiry dates")
    if (~valid_types).any():
        raise SystemExit("Unknown option type values found")
    if t.isna().any():
        raise SystemExit("Invalid observation timestamps found")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python research/validate_options.py <options.parquet>")
    main(sys.argv[1])
