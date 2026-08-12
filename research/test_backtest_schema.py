from __future__ import annotations

import pandas as pd

from backtest import normalize_type, parse_expiry_from_symbol, parse_time


def test_symbol_expiry_is_08_utc():
    s = pd.Series(["BTC-230915-25000-C", "BTC-230915-25000-P"])
    out = parse_expiry_from_symbol(s)
    assert out.iloc[0] == pd.Timestamp("2023-09-15 08:00:00", tz="UTC")
    assert out.iloc[1] == pd.Timestamp("2023-09-15 08:00:00", tz="UTC")


def test_date_plus_hour_builds_utc_time():
    df = pd.DataFrame({"date": ["2023-05-18"], "hour": [13]})
    out = parse_time(df)
    assert out.iloc[0] == pd.Timestamp("2023-05-18 13:00:00", tz="UTC")


def test_option_type_is_explicit():
    out = normalize_type(pd.Series(["CALL", "C", "PUT", "P", "CASH"]))
    assert out.iloc[:4].tolist() == [True, True, False, False]
    assert pd.isna(out.iloc[4])
