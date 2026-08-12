from __future__ import annotations

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ALIASES = {
    'symbol':['symbol','option_symbol'], 'strike':['strike_price','strike'],
    'expiry':['expiry_date','expiration_date','expiry','expiry_time'],
    'type':['side','option_type','type'], 'price':['mark_price','markPrice','last_price','close'],
    'underlying':['underlying_price','index_price','underlyingIndexPrice','underlying'],
    'iv':['mark_iv','markIV','implied_volatility','iv'], 'time':['timestamp','time','open_time']
}

def pick(df, names):
    lower={str(c).lower():c for c in df.columns}
    for n in names:
        if n.lower() in lower:return lower[n.lower()]
    return None

def bs_delta(S,K,T,sigma,call):
    if min(S,K,T,sigma)<=0:return np.nan
    d1=(math.log(S/K)+(sigma*sigma/2)*T)/(sigma*math.sqrt(T))
    return float(norm.cdf(d1) if call else norm.cdf(d1)-1)

def load(path):
    df=pd.read_parquet(path)
    cols={k:pick(df,v) for k,v in ALIASES.items()}
    missing=[k for k,v in cols.items() if v is None and k in ('symbol','strike','expiry','type','price','underlying','time')]
    if missing: raise RuntimeError('Cannot map required Binance fields: '+', '.join(missing)+'; columns='+','.join(map(str,df.columns)))
    x=pd.DataFrame({k:df[v] for k,v in cols.items() if v})
    x['time']=pd.to_datetime(x['time'],unit='ms',errors='coerce')
    # Some archives use microseconds for timestamps.
    bad=x['time'].isna() | (x['time']<pd.Timestamp('2020-01-01'))
    if bad.any(): x.loc[bad,'time']=pd.to_datetime(df.loc[bad,cols['time']],unit='us',errors='coerce')
    x['expiry']=pd.to_datetime(x['expiry'],unit='ms',errors='coerce')
    bad=x['expiry'].isna() | (x['expiry']<pd.Timestamp('2020-01-01'))
    if bad.any(): x.loc[bad,'expiry']=pd.to_datetime(df.loc[bad,cols['expiry']],unit='us',errors='coerce')
    x['call']=x['type'].astype(str).str.upper().str.contains('CALL|C')
    for c in ['strike','price','underlying','iv']: x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x.dropna(subset=['symbol','strike','expiry','price','underlying','time'])
    return x

def nearest_exit(g, horizon):
    g=g.sort_values('time').reset_index(drop=True)
    targets=g['time']+pd.Timedelta(days=horizon)
    # merge_asof is used by group in the main loop; this helper documents intended tolerance.
    return targets

def run(df,cfg,out):
    out.mkdir(parents=True,exist_ok=True)
    fee=cfg['costs']['fee_bps']/10000; spread=cfg['costs']['spread_bps']/10000; slip=cfg['costs']['slippage_bps']/10000
    results=[]
    # Work option-by-option. To control memory, only retain rows needed for each contract.
    for symbol,g in df.groupby('symbol',sort=False):
        g=g.sort_values('time').copy()
        if len(g)<2: continue
        for horizon in cfg['horizons_days']:
            # A tolerance of +/- 30 minutes accommodates small archive timestamp differences.
            target=g[['time','price','underlying','strike','expiry','call','iv']].copy()
            target['entry_time']=target['time']
            target['exit_target']=target['time']+pd.Timedelta(days=horizon)
            exits=g[['time','price','underlying']].rename(columns={'time':'exit_time','price':'exit_price','underlying':'exit_underlying'})
            m=pd.merge_asof(target.sort_values('exit_target'),exits.sort_values('exit_time'),left_on='exit_target',right_on='exit_time',direction='nearest',tolerance=pd.Timedelta(minutes=30))
            m=m.dropna(subset=['exit_price'])
            m=m[m['exit_time']<m['expiry']]
            if m.empty: continue
            for _,r in m.iterrows():
                S=float(r.underlying); K=float(r.strike); T=max((r.expiry-r.entry_time).total_seconds()/31557600,1e-8)
                iv=float(r.iv) if pd.notna(r.iv) and float(r.iv)>0 else np.nan
                if not np.isfinite(iv): continue
                d=bs_delta(S,K,T,iv,bool(r.call))
                if not np.isfinite(d): continue
                opt_pnl=float(r.exit_price-r.price)
                spot_move=float(r.exit_underlying-r.underlying)
                strategies={'static_1to1':1.0 if r.call else -1.0,'delta_50':0.5*d,'delta_dynamic':d,'delta_threshold_10':d}
                for hedge,h in strategies.items():
                    # Hedge is opposite the option delta. Dynamic hedge uses entry delta for this EOH-to-EOH test;
                    # threshold strategy only hedges when delta changes by >=10%, approximated from endpoint delta.
                    if hedge=='delta_threshold_10':
                        T2=max((r.expiry-r.exit_time).total_seconds()/31557600,1e-8)
                        d2=bs_delta(float(r.exit_underlying),K,T2,iv,bool(r.call))
                        hedge_qty= -d if (np.isfinite(d2) and abs(d2-d)>=0.10) else 0.0
                    else: hedge_qty=-h
                    gross=opt_pnl + hedge_qty*spot_move
                    traded=max(abs(float(r.price)),1e-12)+abs(hedge_qty)*max(S,1e-12)
                    cost=traded*(fee+spread+slip)
                    results.append({'symbol':symbol,'entry_time':r.entry_time,'exit_time':r.exit_time,'horizon_days':horizon,'option_type':'CALL' if r.call else 'PUT','hedge':hedge,'iv':iv,'delta':d,'gross_pnl':gross,'cost':cost,'net_pnl':gross-cost,'entry_price':r.price,'exit_price':r.exit_price})
    trades=pd.DataFrame(results)
    if trades.empty: raise RuntimeError('No valid backtest observations. Inspect validation output/schema.')
    trades.to_csv(out/'trades.csv',index=False)
    summary=(trades.groupby(['horizon_days','option_type','hedge']).agg(trades=('net_pnl','size'),net_pnl=('net_pnl','sum'),mean_pnl=('net_pnl','mean'),win_rate=('net_pnl',lambda x:(x>0).mean()),pnl_std=('net_pnl','std')).reset_index())
    summary.to_csv(out/'strategy_comparison.csv',index=False)
    print(summary.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(load(a.input),json.loads(Path(a.config).read_text()),Path(a.output))
