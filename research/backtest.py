from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL_RE = re.compile(r"^BTC-(\d{6})-([0-9.]+)-(C|P)$", re.IGNORECASE)


def first_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def parse_time(df: pd.DataFrame) -> pd.Series:
    explicit = first_column(df, ["timestamp", "time", "open_time"])
    if explicit:
        s = pd.to_numeric(df[explicit], errors="coerce")
        out = pd.to_datetime(s, unit="ms", errors="coerce", utc=True)
        bad = out.isna() | (out < pd.Timestamp("2020-01-01", tz="UTC"))
        if bad.any():
            out.loc[bad] = pd.to_datetime(s.loc[bad], unit="us", errors="coerce", utc=True)
        return out
    date_col = first_column(df, ["date"])
    hour_col = first_column(df, ["hour"])
    if not date_col or not hour_col:
        raise RuntimeError("Binance EOH dataset requires timestamp/time or date + hour")
    date_part = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.normalize()
    hour = pd.to_numeric(df[hour_col], errors="coerce")
    return date_part + pd.to_timedelta(hour, unit="h")


def parse_expiry_from_symbol(symbols: pd.Series) -> pd.Series:
    parts = symbols.astype(str).str.extract(SYMBOL_RE)
    expiry = pd.to_datetime(parts[0], format="%y%m%d", errors="coerce", utc=True)
    return expiry + pd.Timedelta(hours=8)


def parse_expiry(df: pd.DataFrame, symbols: pd.Series) -> pd.Series:
    explicit = first_column(df, ["expiry_date", "expiration_date", "expiry", "expiry_time"])
    if explicit:
        s = df[explicit]
        numeric = pd.to_numeric(s, errors="coerce")
        out = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
        bad = out.isna() | (out < pd.Timestamp("2020-01-01", tz="UTC"))
        if bad.any():
            out.loc[bad] = pd.to_datetime(s.loc[bad], errors="coerce", utc=True)
            bad2 = out.isna()
            if bad2.any():
                out.loc[bad2] = pd.to_datetime(numeric.loc[bad2], unit="us", errors="coerce", utc=True)
        return out
    return parse_expiry_from_symbol(symbols)


def normalize_type(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.upper().str.strip()
    return t.map({"CALL": True, "C": True, "PUT": False, "P": False})


def load(path: str | Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    symbol_col = first_column(raw, ["symbol", "option_symbol"])
    strike_col = first_column(raw, ["strike", "strike_price"])
    type_col = first_column(raw, ["type", "option_type", "side"])
    price_col = first_column(raw, ["mark_price", "markPrice", "price"])
    underlying_col = first_column(raw, ["underlying", "underlying_price", "index_price"])
    iv_col = first_column(raw, ["mark_iv", "markIV", "iv", "implied_volatility"])
    delta_col = first_column(raw, ["delta"])
    required = {"symbol": symbol_col, "strike": strike_col, "type": type_col, "price": price_col, "underlying": underlying_col}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(f"Cannot map required Binance fields: {', '.join(missing)}; columns={list(raw.columns)}")
    x = pd.DataFrame({
        "symbol": raw[symbol_col].astype(str),
        "strike": pd.to_numeric(raw[strike_col], errors="coerce"),
        "call": normalize_type(raw[type_col]),
        "price": pd.to_numeric(raw[price_col], errors="coerce"),
        "underlying": pd.to_numeric(raw[underlying_col], errors="coerce"),
        "time": parse_time(raw),
        "expiry": parse_expiry(raw, raw[symbol_col].astype(str)),
    })
    x["iv"] = pd.to_numeric(raw[iv_col], errors="coerce") if iv_col else np.nan
    x["delta"] = pd.to_numeric(raw[delta_col], errors="coerce") if delta_col else np.nan
    x = x[x["call"].notna()].copy()
    x["call"] = x["call"].astype(bool)
    x = x.dropna(subset=["symbol", "strike", "price", "underlying", "time", "expiry"])
    x = x[x["time"] < x["expiry"]]
    return x.sort_values(["symbol", "time"]).drop_duplicates(["symbol", "time"], keep="last").reset_index(drop=True)


def run(df: pd.DataFrame, cfg: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cost_rate = sum(float(cfg["costs"].get(k, 0.0)) for k in ("fee_bps", "spread_bps", "slippage_bps")) / 10000.0
    horizons = [int(x) for x in cfg.get("horizons_days", [7, 14, 30])]
    hedges = ["static_1to1", "delta_50", "delta_entry"]
    rows: list[pd.DataFrame] = []

    for symbol, g in df.groupby("symbol", sort=False):
        g = g.sort_values("time").reset_index(drop=True)
        if len(g) < 2:
            continue
        exits = g[["time", "price", "underlying"]].rename(columns={"time": "exit_time", "price": "exit_price", "underlying": "exit_underlying"})
        for horizon in horizons:
            target = g[["time", "price", "underlying", "strike", "expiry", "call", "iv", "delta"]].copy()
            target["exit_target"] = target["time"] + pd.Timedelta(days=horizon)
            m = pd.merge_asof(target.sort_values("exit_target"), exits.sort_values("exit_time"), left_on="exit_target", right_on="exit_time", direction="nearest", tolerance=pd.Timedelta(minutes=30))
            m = m.dropna(subset=["exit_price", "exit_underlying"])
            m = m[m["exit_time"] < m["expiry"]]
            if m.empty:
                continue
            m["option_type"] = np.where(m["call"], "CALL", "PUT")
            m["option_pnl"] = m["price"] - m["exit_price"]
            m["spot_move"] = m["exit_underlying"] - m["underlying"]
            for mode in hedges:
                if mode == "static_1to1":
                    hedge = np.where(m["call"], 1.0, -1.0)
                    valid = np.ones(len(m), dtype=bool)
                elif mode == "delta_50":
                    hedge = np.where(m["call"], 0.5, -0.5)
                    valid = np.ones(len(m), dtype=bool)
                else:
                    hedge = pd.to_numeric(m["delta"], errors="coerce").to_numpy()
                    valid = np.isfinite(hedge) & (np.abs(hedge) <= 1.5)
                if not valid.any():
                    continue
                r = m.loc[valid, ["symbol", "time", "exit_time", "strike", "expiry", "option_type", "price", "exit_price", "underlying", "exit_underlying", "iv", "delta"]].copy()
                h = np.asarray(hedge)[valid]
                r["horizon_days"] = horizon
                r["hedge"] = mode
                r["hedge_qty"] = h
                r["option_pnl"] = r["price"] - r["exit_price"]
                r["futures_pnl"] = h * (r["exit_underlying"] - r["underlying"])
                r["gross_pnl"] = r["option_pnl"] + r["futures_pnl"]
                r["estimated_cost"] = (r["price"].abs() + np.abs(h) * r["underlying"].abs()) * cost_rate
                r["net_pnl"] = r["gross_pnl"] - r["estimated_cost"]
                rows.append(r)

    if not rows:
        raise RuntimeError("No valid backtest observations after schema/expiry filtering")
    trades = pd.concat(rows, ignore_index=True)
    trades.to_csv(out / "trades.csv", index=False)

    def profit_factor(s: pd.Series) -> float:
        gains = s[s > 0].sum()
        losses = -s[s < 0].sum()
        return float(gains / losses) if losses > 0 else float("inf")

    summary = trades.groupby(["horizon_days", "option_type", "hedge"], as_index=False).agg(
        trades=("net_pnl", "size"), net_pnl=("net_pnl", "sum"), mean_pnl=("net_pnl", "mean"), median_pnl=("net_pnl", "median"),
        win_rate=("net_pnl", lambda x: float((x > 0).mean())), pnl_std=("net_pnl", "std"))
    summary["profit_factor"] = [profit_factor(trades.loc[(trades.horizon_days == r.horizon_days) & (trades.option_type == r.option_type) & (trades.hedge == r.hedge), "net_pnl"]) for r in summary.itertuples()]
    summary.to_csv(out / "strategy_comparison.csv", index=False)
    metadata = {
        "rows_loaded": int(len(df)), "trade_rows": int(len(trades)), "hedges_tested": hedges, "horizons_days": horizons,
        "cost_model": "fee + spread + slippage placeholder; funding not modeled in baseline",
        "iv_rv_filter": "not implemented in baseline",
        "threshold_dynamic_hedging": "deferred until hourly hedge-path simulation is implemented",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(load(a.input), json.loads(Path(a.config).read_text()), Path(a.output))
