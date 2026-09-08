"""Unified market calendar with deterministic and symbol-specific events."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import calendar, os, json

def _last_friday(year, month):
    day=calendar.monthrange(year,month)[1]
    d=datetime(year,month,day,tzinfo=timezone.utc)
    return d-timedelta(days=(d.weekday()-4)%7)

def events(symbols=None, months=3):
    now=datetime.now(timezone.utc); end=now+timedelta(days=31*months); out=[]
    cursor=datetime(now.year,now.month,1,tzinfo=timezone.utc)
    while cursor<=end:
        expiry=_last_friday(cursor.year,cursor.month)
        if expiry>=now-timedelta(days=1): out.append({"date":expiry.date().isoformat(),"time":"08:00 UTC (estimado)","category":"derivatives","impact":"high","title":"Vencimento mensal Deribit BTC/ETH — estimativa","source":"regra Deribit: última sexta-feira","estimated":True})
        cursor=datetime(cursor.year+(cursor.month==12),cursor.month%12+1,1,tzinfo=timezone.utc)
    # COT publication is normally Friday; generate forthcoming occurrences.
    d=now
    while d<=end:
        if d.weekday()==4: out.append({"date":d.date().isoformat(),"time":"após fechamento (estimado)","category":"commodities","impact":"medium","title":"Janela esperada CFTC Commitments of Traders","source":"calendário semanal; feriados podem alterar","estimated":True})
        d+=timedelta(days=1)
    for sym in symbols or []:
        try:
            import yfinance as yf
            cal=yf.Ticker(sym).calendar or {}; dates=cal.get("Earnings Date") or []
            if not isinstance(dates,(list,tuple)): dates=[dates]
            for x in dates:
                dt=x.to_pydatetime() if hasattr(x,"to_pydatetime") else x
                if dt: out.append({"date":str(dt)[:10],"time":"a confirmar","category":"earnings","impact":"high","title":f"Resultado {sym}","source":"Yahoo Finance"})
        except Exception: pass
    try:
        for x in json.loads(os.getenv("MACRO_EVENTS_JSON","[]")):
            if isinstance(x,dict) and x.get("date"): out.append(x)
    except ValueError: pass
    out.sort(key=lambda x:(x.get("date",""),x.get("time","")))
    return {"events":out,"generated_at":int(now.timestamp()*1000),"note":"Eventos macro específicos podem ser adicionados por MACRO_EVENTS_JSON."}
