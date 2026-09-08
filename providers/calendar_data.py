"""Calendário de mercado: eventos determinísticos + earnings por símbolo."""
from __future__ import annotations

import calendar
import json
import os
from datetime import datetime, timedelta, timezone


def _last_friday(year: int, month: int) -> datetime:
    day = calendar.monthrange(year, month)[1]
    d = datetime(year, month, day, tzinfo=timezone.utc)
    return d - timedelta(days=(d.weekday() - 4) % 7)


def _deribit_expiries(now: datetime, end: datetime) -> list[dict]:
    """Vencimento mensal Deribit: última sexta-feira de cada mês."""
    out = []
    cursor = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    while cursor <= end:
        expiry = _last_friday(cursor.year, cursor.month)
        if expiry >= now - timedelta(days=1):
            out.append({
                "date": expiry.date().isoformat(),
                "time": "08:00 UTC (estimado)",
                "category": "derivatives", "impact": "high",
                "title": "Vencimento mensal Deribit BTC/ETH — estimativa",
                "source": "regra Deribit: última sexta-feira", "estimated": True,
            })
        cursor = datetime(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1,
                          tzinfo=timezone.utc)
    return out


def _cot_windows(now: datetime, end: datetime) -> list[dict]:
    """CFTC COT sai às sextas; feriados podem deslocar."""
    out, d = [], now
    while d <= end:
        if d.weekday() == 4:
            out.append({
                "date": d.date().isoformat(), "time": "após fechamento (estimado)",
                "category": "commodities", "impact": "medium",
                "title": "Janela esperada CFTC Commitments of Traders",
                "source": "calendário semanal; feriados podem alterar", "estimated": True,
            })
        d += timedelta(days=1)
    return out


def _earnings(symbols) -> list[dict]:
    import yfinance as yf

    out = []
    for sym in symbols:
        try:
            cal = yf.Ticker(sym).calendar or {}
            dates = cal.get("Earnings Date") or []
            if not isinstance(dates, (list, tuple)):
                dates = [dates]
            for x in dates:
                dt = x.to_pydatetime() if hasattr(x, "to_pydatetime") else x
                if dt:
                    out.append({"date": str(dt)[:10], "time": "a confirmar",
                                "category": "earnings", "impact": "high",
                                "title": f"Resultado {sym}", "source": "Yahoo Finance"})
        except Exception:
            continue  # sem earnings para o símbolo não invalida o calendário
    return out


def events(symbols=None, months: int = 3) -> dict:
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=31 * months)

    out = _deribit_expiries(now, end) + _cot_windows(now, end) + _earnings(symbols or [])
    try:
        out += [x for x in json.loads(os.getenv("MACRO_EVENTS_JSON", "[]"))
                if isinstance(x, dict) and x.get("date")]
    except ValueError:
        pass

    out.sort(key=lambda x: (x.get("date", ""), x.get("time", "")))
    return {
        "events": out,
        "generated_at": int(now.timestamp() * 1000),
        "note": "Eventos macro específicos podem ser adicionados por MACRO_EVENTS_JSON.",
    }
