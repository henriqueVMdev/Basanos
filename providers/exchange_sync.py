"""
Sincronização de operações reais das corretoras para o Diário.

Puxa os trades fechados (fills) e as taxas pagas de cada corretora via CCXT,
normaliza para o formato do journal e devolve uma lista pronta para mesclar.

Credenciais são lidas de variáveis de ambiente (.env na raiz do projeto):

    BINGX_API_KEY,        BINGX_API_SECRET
    OKX_API_KEY,          OKX_API_SECRET,        OKX_PASSWORD
    HYPERLIQUID_WALLET,   HYPERLIQUID_PRIVATE_KEY

SimpleFX não tem suporte no CCXT e será integrado em etapa futura.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import ccxt
from dotenv import load_dotenv

# Carrega o .env da raiz do projeto (idempotente).
load_dotenv(Path(__file__).parents[1] / ".env")

# Corretoras suportadas via CCXT e as env vars que cada uma exige.
SUPPORTED = ("bingx", "okx", "hyperliquid")

# Chaves comuns onde o PnL realizado costuma aparecer no payload bruto (info).
_PNL_KEYS = (
    "realizedPnl", "realisedPnl", "realized_pnl", "closedPnl",
    "closPnl", "pnl", "rpl", "profit",
)


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _build_exchange(name: str):
    """Instancia o cliente CCXT autenticado, ou (None, motivo) se faltar chave."""
    name = name.lower()
    if name == "bingx":
        key, sec = os.getenv("BINGX_API_KEY"), os.getenv("BINGX_API_SECRET")
        if not key or not sec:
            return None, "BINGX_API_KEY/BINGX_API_SECRET ausentes no .env"
        return ccxt.bingx({
            "apiKey": key, "secret": sec, "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }), None

    if name == "okx":
        key = os.getenv("OKX_API_KEY")
        sec = os.getenv("OKX_API_SECRET")
        pwd = os.getenv("OKX_PASSWORD")
        if not key or not sec or not pwd:
            return None, "OKX_API_KEY/OKX_API_SECRET/OKX_PASSWORD ausentes no .env"
        return ccxt.okx({
            "apiKey": key, "secret": sec, "password": pwd, "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }), None

    if name == "hyperliquid":
        wallet = os.getenv("HYPERLIQUID_WALLET")
        pk = os.getenv("HYPERLIQUID_PRIVATE_KEY")
        if not wallet:
            return None, "HYPERLIQUID_WALLET ausente no .env"
        cfg = {"walletAddress": wallet, "enableRateLimit": True}
        if pk:
            cfg["privateKey"] = pk
        return ccxt.hyperliquid(cfg), None

    return None, f"corretora '{name}' não suportada via CCXT"


def _extract_pnl(trade: dict):
    """Tenta extrair o PnL realizado do payload bruto (varia por corretora)."""
    info = trade.get("info") or {}
    for k in _PNL_KEYS:
        if k in info and info[k] not in (None, "", "0", 0):
            return _to_float(info[k])
    # Algumas corretoras já trazem 'pnl' no nível normalizado.
    if trade.get("pnl") is not None:
        return _to_float(trade["pnl"])
    return None


def _normalize(name: str, trade: dict) -> dict:
    """Converte um fill do CCXT no formato de trade do journal."""
    fee_obj = trade.get("fee") or {}
    fee_cost = abs(_to_float(fee_obj.get("cost")))
    # Algumas corretoras devolvem múltiplas fees em 'fees'.
    if not fee_cost and trade.get("fees"):
        fee_cost = sum(abs(_to_float(f.get("cost"))) for f in trade["fees"])

    pnl = _extract_pnl(trade)
    # amount do journal = PnL realizado quando disponível; senão, o custo do fill.
    if pnl is not None:
        amount = abs(pnl)
        result = "gain" if pnl >= 0 else "loss"
    else:
        amount = 0.0
        result = "gain"

    ts = trade.get("timestamp")
    date_str = ""
    if ts:
        date_str = time.strftime("%Y-%m-%d", time.gmtime(ts / 1000))

    return {
        "external_id": f"{name}:{trade.get('id') or trade.get('order') or ts}",
        "source": "exchange",
        "exchange": name,
        "date": date_str,
        "strategy": name,                       # agrupa por corretora em "Por Estratégia"
        "asset": trade.get("symbol") or "",
        "side": trade.get("side") or "",
        "result": result,
        "amount": round(amount, 2),
        "pnl": round(pnl, 4) if pnl is not None else None,
        "fee": round(fee_cost, 4),
        "fee_currency": fee_obj.get("currency") or "",
        "notes": "",
    }


def fetch_exchange_trades(name: str, since_days: int = 30, limit: int = 200):
    """
    Puxa os trades de uma corretora. Devolve (lista_normalizada, warning|None).

    since_days: janela em dias para trás (algumas corretoras limitam o histórico).
    """
    ex, reason = _build_exchange(name)
    if ex is None:
        return [], reason

    since = int((time.time() - since_days * 86400) * 1000)
    try:
        raw = ex.fetch_my_trades(since=since, limit=limit)
    except ccxt.AuthenticationError as e:
        return [], f"{name}: autenticação falhou ({e})"
    except Exception as e:
        # Várias corretoras exigem 'symbol' no fetch_my_trades sem ele.
        return [], f"{name}: erro ao buscar trades ({type(e).__name__}: {e})"

    return [_normalize(name, t) for t in raw], None


def sync_all(exchanges=None, since_days: int = 30):
    """
    Sincroniza todas as corretoras configuradas.

    Retorna dict: {"trades": [...], "warnings": [...], "by_exchange": {name: count}}.
    """
    exchanges = [e.lower() for e in (exchanges or SUPPORTED)]
    all_trades, warnings, counts = [], [], {}
    for name in exchanges:
        if name not in SUPPORTED:
            warnings.append(f"{name}: não suportado via CCXT (ignorado)")
            counts[name] = 0
            continue
        trades, warn = fetch_exchange_trades(name, since_days=since_days)
        if warn:
            warnings.append(warn)
        all_trades.extend(trades)
        counts[name] = len(trades)
    return {"trades": all_trades, "warnings": warnings, "by_exchange": counts}
