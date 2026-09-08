# -*- coding: utf-8 -*-
"""
Intelligence Signal — segue o sinal da Central de Inteligência (POSICIONAL)
===========================================================================
Forward test do motor de cruzamento: COMPRA -> long, VENDA -> short,
NEUTRO -> flat. Sizing fixo (exposure constante), entrada a MERCADO na
abertura do candle seguinte, saída quando o rótulo muda (exit_on_flip).
Sem TP/SL/time-stop: a tese é o próprio sinal.

params:
- symbol: ativo consultado na central (default: o símbolo do deployment
  não chega ao signal(); configure ex. {"symbol": "BTC"})
- exposure: fração fixa do equity por posição (default 1.0)
- min_confidence: ignora COMPRA/VENDA com concordância abaixo disso (default 0)

Honestidade:
- Forward puro: analyze() usa dados AO VIVO. Em replay de candles perdidos
  (PC desligado) o sinal aplicado é o ATUAL, não o histórico daquele dia —
  gaps longos aproximam. O forward válido é o processo ligado dia a dia.
- Falha transitória de fonte NÃO fecha posição: mantém a última leitura.

Módulo só de automação (signal()); não expõe run() para o frontend.
"""

NAME = "Intelligence Signal"
DESCRIPTION = ("Segue COMPRA/VENDA da Central de Inteligência com sizing fixo "
               "(posicional, forward test do motor de cruzamento)")

_last_side: dict = {}   # symbol -> último lado válido (segura falha de fonte)


def signal(df, params: dict):
    p = params or {}
    sym = (p.get("symbol") or "BTC").upper()
    if df is None or len(df) < 5:
        return None
    try:
        from providers import intelligence_data
        r = intelligence_data.analyze(sym)
        sig = r["signal"]
        side = 1 if sig["label"] == "COMPRA" else -1 if sig["label"] == "VENDA" else 0
        if side and (sig["confidence"] or 0) < float(p.get("min_confidence", 0)):
            side = 0
        _last_side[sym] = side
    except Exception:
        side = _last_side.get(sym)  # fonte fora do ar: não fecha posição à toa
        if side is None:
            return None
    if not side:
        return None
    return {
        "side": side,
        "type": "market",
        "price": None,
        "valid_bars": 1,
        "tp_pct": None,
        "sl_pct": None,
        "max_bars": None,
        "fill_rule": "open",
        "exposure": float(p.get("exposure", 1.0)),
        "exit_on_flip": True,
    }
