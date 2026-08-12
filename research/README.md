# BTC Options Research

Research/backtesting scaffold for historical Binance BTC options.

## Scope
- Historical period: 2023-05-18 through 2023-10-23
- Underlying: BTCUSDT
- Primary dataset: Binance Options EOHSummary
- Strategy variants: static, partial, dynamic and threshold delta hedging
- Horizons: 7D, 14D, 30D
- Call and put variants
- IV/RV premium filters: none, +5%, +10%, +20%
- Transaction costs: configurable fees, spread, slippage and funding

## Data source
Binance Public Data (`data.binance.vision`). Binance documents daily/monthly public market-data archives and programmatic download support in the repository README.

## Reproducibility
Run `python -m research.btc_options download` to download the configured archives, then `python -m research.btc_options backtest` to produce results. Raw downloaded archives are intentionally gitignored.
