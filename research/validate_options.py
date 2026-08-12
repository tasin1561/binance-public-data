from __future__ import annotations
import sys
import pandas as pd

p = sys.argv[1]
df = pd.read_parquet(p)
print('rows:', len(df))
print('columns:', ', '.join(map(str, df.columns)))
if df.empty:
    raise SystemExit('Dataset is empty')
# Keep validation structural; Binance has changed option archive schemas over time.
required_any = [
    {'symbol', 'underlying'},
    {'mark_price'},
    {'strike_price', 'strike'},
    {'expiry_date', 'expiration_date', 'expiry'},
]
for group in required_any:
    if not (set(map(str.lower, df.columns)) & set(group)):
        print('warning: none of expected fields found for', group)
