"""US/systemic liquidity snapshot sourced from FRED CSV endpoints."""
from __future__ import annotations
from io import StringIO
import time, requests, pandas as pd

SERIES={"fed_balance_sheet":("WALCL","Fed balance sheet","USD millions"),"reverse_repo":("RRPONTSYD","Reverse repo","USD billions"),"treasury_account":("WTREGEN","Treasury General Account","USD millions"),"high_yield_spread":("BAMLH0A0HYM2","US high-yield spread","percent"),"vix":("VIXCLS","VIX","index"),"broad_dollar":("DTWEXBGS","Broad dollar index","index")}
_CACHE=None
def snapshot():
    global _CACHE
    if _CACHE and time.time()-_CACHE[0]<3600:return _CACHE[1]
    rows=[]
    for key,(sid,label,unit) in SERIES.items():
        try:
            r=requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",timeout=25);r.raise_for_status()
            df=pd.read_csv(StringIO(r.text)); vals=pd.to_numeric(df.iloc[:,1],errors='coerce').dropna()
            rows.append({"id":key,"series":sid,"label":label,"value":float(vals.iloc[-1]),"change_30":float(vals.iloc[-1]-vals.iloc[-min(31,len(vals))]),"unit":unit,"status":"ok"})
        except Exception as e: rows.append({"id":key,"series":sid,"label":label,"status":"error","error":str(e)[:80]})
    out={"series":rows,"source":"Federal Reserve Economic Data (FRED)","ts":int(time.time()*1000)};_CACHE=(time.time(),out);return out
