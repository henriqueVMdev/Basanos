"""
Estratégia MM9 / TrapM-LW — reprodução do indicador fechado do TradingView
===========================================================================
Engenharia reversa validada contra o Strategy Tester (BTC 15m Bybit,
242 posições / 374 legs, 2026-07-08 — research/analyze_trapm*.py):

- VIÉS: Close vs média LENTA (SMA40) — acima só long, abaixo só short
  (bateu em 100% das 242 posições).
- GATILHO: candle de ENGOLFO clássico (230/242) que TOCA a média rápida
  (EMA8): long exige Low <= rápida; short exige High >= rápida (241/242).
- ENTRADA: ordem STOP no extremo do candle de gatilho (204/242 fills no
  candle seguinte; a ordem PERSISTE armada — fills observados até 8
  candles depois, sempre no nível do gatilho original). Cancela quando o
  viés inverte; um novo gatilho substitui o nível.
- FILTRO MIN GAIN (update recente do TrapM): risco entrada->stop >= 0,5%
  (piso seco observado: min 0,504%, p5 0,511%) — elimina ruído.
- STOP estrutural: extremo dos 4 candles até o gatilho (93% com k=4).
- SAÍDAS em duas pernas:
    TP1 = 1R fecha 30% (limite, SEM stop próprio — 98 fills);
    TP2 = 2R fecha 70% (limite) com stop estrutural na perna (52+45 fills);
    Slow MA Stop: Close cruza a SMA40 contra a posição -> fecha TUDO que
    restar a MERCADO no close do candle (178 fills, preço == close do
    candle do cruzamento; vale já no candle do fill — 13 casos).
- SIZING: risco por trade (qty proporcional a risco% / distância do stop).

Semântica intrabar do candle do fill (entrada STOP preenche a caminho do
extremo): TP alcançável no mesmo candle (o extremo vem depois do fill);
SL só se o candle fechou CONTRA a posição (trajetória verde O-L-H-C /
vermelha O-H-L-C, mesma regra validada na MM9 Pullback Maker).

Espelha o contrato de retorno de strategies/depaula.py (metrics,
equity_curve, trades, drawdown, chart), incluindo os modos _fast e _charts.
"""

import sys
import os
import math
import numpy as np
import pandas as pd

# Garante que o diretório raiz do projeto está no sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.backtesting import calc_ma, Trade, brt_hour

NAME = "MM9 / TrapM — Engolfo + 2 Alvos"
DESCRIPTION = (
    "Reprodução do TrapM-LW: viés por Close vs média lenta, engolfo tocando "
    "a média rápida, entrada stop no rompimento (ordem persistente), filtro "
    "de ganho mínimo, TP1 1R (30%) + TP2 2R (70%) com stop estrutural e "
    "saída geral no cruzamento da média lenta."
)

CONFIG_SCHEMA = [
    {
        "title": "Médias",
        "fields": [
            {"key": "ma_fast_type", "label": "Tipo Rápida", "type": "select",
             "default": "EMA", "options": ["SMA", "EMA", "HMA", "RMA", "WMA"]},
            {"key": "ma_fast_len", "label": "Período Rápida", "type": "number",
             "default": 6, "min": 2, "max": 200, "step": 1},
            {"key": "ma_slow_type", "label": "Tipo Lenta", "type": "select",
             "default": "SMA", "options": ["SMA", "EMA", "HMA", "RMA", "WMA"]},
            {"key": "ma_slow_len", "label": "Período Lenta", "type": "number",
             "default": 40, "min": 3, "max": 500, "step": 1},
        ],
    },
    {
        "title": "Gatilho",
        "fields": [
            {"key": "use_engulfing", "label": "Usar Engolfo", "type": "checkbox",
             "default": True},
            {"key": "use_pfr", "label": "Usar PFR", "type": "checkbox",
             "default": False},
            {"key": "min_gain_pct", "label": "Ganho mínimo (risco %)", "type": "number",
             "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1},
        ],
    },
    {
        "title": "Stop / Alvos",
        "fields": [
            {"key": "stop_lookback", "label": "Extremos p/ Stop (N candles)",
             "type": "number", "default": 4, "min": 1, "max": 50, "step": 1},
            {"key": "tick_size", "label": "Tick (folga do stop)", "type": "number",
             "default": 0.1, "min": 0.0, "step": 0.01},
            {"key": "tp1_r", "label": "Alvo 1 (x risco)", "type": "number",
             "default": 1.0, "min": 0.1, "step": 0.1},
            {"key": "tp1_pct", "label": "Realizar no Alvo 1 (%)", "type": "number",
             "default": 30.0, "min": 0, "max": 100, "step": 5},
            {"key": "tp2_r", "label": "Alvo 2 (x risco)", "type": "number",
             "default": 2.0, "min": 0.1, "step": 0.1},
        ],
    },
    {
        "title": "Alavancagem / Sizing",
        "fields": [
            {"key": "sizing_mode", "label": "Modo de Sizing", "type": "select",
             "default": "Risco por trade",
             "options": ["Alavancagem fixa", "Quantidade fixa", "Risco por trade"]},
            {"key": "leverage", "label": "Alavancagem máx (x)", "type": "number",
             "default": 2.0, "min": 1.0, "max": 125.0, "step": 0.5},
            {"key": "margin_pct", "label": "Margem por Trade (% do capital)",
             "type": "number", "default": 100.0, "min": 1.0, "max": 100.0, "step": 5.0},
            {"key": "fixed_qty", "label": "Quantidade Fixa (unidades)", "type": "number",
             "default": 0.1, "min": 0.0, "step": 0.01},
            {"key": "risk_per_trade", "label": "Risco por Trade (%)", "type": "number",
             "default": 1.0, "min": 0.1, "max": 100, "step": 0.1},
        ],
    },
]


OPTIMIZER_GRIDS = {
    "rapido": {
        "ma_fast_len":   [6, 8, 12],
        "ma_slow_len":   [30, 40, 60],
        "stop_lookback": [2, 4, 6],
        "tp1_r":         [1.0],
        "tp2_r":         [1.5, 2.0, 3.0],
        "min_gain_pct":  [0.0, 0.3, 0.5, 0.8],
    },
    "completo": {
        "ma_fast_type":  ["EMA"],
        "ma_fast_len":   [5, 6, 8, 10, 12],
        "ma_slow_type":  ["SMA", "EMA"],
        "ma_slow_len":   [21, 30, 40, 50, 80],
        "stop_lookback": [2, 3, 4, 6, 8],
        "tp1_r":         [0.5, 1.0, 1.5],
        "tp1_pct":       [30, 50],
        "tp2_r":         [1.5, 2.0, 3.0],
        "min_gain_pct":  [0.0, 0.3, 0.5, 0.8, 1.2],
    },
}


def is_valid_config(params):
    """Remove combinacoes que nao fazem sentido para esta estrategia."""
    if int(params.get("ma_fast_len", 8)) >= int(params.get("ma_slow_len", 40)):
        return False
    if float(params.get("tp1_r", 1.0)) >= float(params.get("tp2_r", 2.0)):
        return False
    if not params.get("use_engulfing", True) and not params.get("use_pfr", False):
        return False
    return True


def prepare_optimizer_params(params):
    """Sem simetrias derivadas nesta estrategia."""
    return params


def _safe(val):
    """Converte NaN/Inf para None (JSON-safe)."""
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _is_engulfing(d, o, c, o1, c1):
    """Engolfo clássico: corpo do candle atual engolfa o corpo anterior + direção."""
    if d == 1:
        return c > o and o <= c1 and c >= o1
    return c < o and o >= c1 and c <= o1


def _is_pfr(d, h, l, h1, l1, c, c1):
    """PFR: rompe a extremidade anterior e fecha de volta."""
    if d == 1:
        return l < l1 and c > c1
    return h > h1 and c < c1


def _compute_sizing(equity, entry, stop, params):
    """Retorna (qty, leverage, exposure) conforme o modo de sizing."""
    equity = equity if equity > 0 else 1.0
    lev = max(float(params.get("leverage", 2.0)), 1.0)
    mode = params.get("sizing_mode", "Risco por trade")

    if mode == "Quantidade fixa":
        qty = max(float(params.get("fixed_qty", 0.0)), 0.0)
        exposure = (qty * entry) / equity
        return qty, lev, exposure

    if mode == "Risco por trade":
        stop_dist_pct = abs(entry - stop) / entry * 100 if entry > 0 else 0.0
        exposure = (float(params.get("risk_per_trade", 1.0)) / stop_dist_pct
                    if stop_dist_pct > 0 else 1.0)
        exposure = min(exposure, lev)
        qty = (exposure * equity) / entry if entry > 0 else 0.0
        return qty, lev, exposure

    # "Alavancagem fixa"
    margin_frac = max(min(float(params.get("margin_pct", 100.0)) / 100.0, 1.0), 0.0)
    exposure = margin_frac * lev
    qty = (exposure * equity) / entry if entry > 0 else 0.0
    return qty, lev, exposure


def _run_backtest_mm9(df: pd.DataFrame, params: dict):
    """Motor barra a barra da spec TrapM. Retorna (bt_df, trades, final_equity)."""
    df = df.copy()

    fast_len = int(params.get("ma_fast_len", 6))
    slow_len = int(params.get("ma_slow_len", 40))
    df["Fast"] = calc_ma(df["Close"], fast_len, params.get("ma_fast_type", "EMA"))
    df["Slow"] = calc_ma(df["Close"], slow_len, params.get("ma_slow_type", "SMA"))

    start_date = params.get("start_date")
    end_date = params.get("end_date")
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    use_engulfing = bool(params.get("use_engulfing", True))
    use_pfr = bool(params.get("use_pfr", False))
    stop_lookback = max(int(params.get("stop_lookback", 4)), 1)
    tick = float(params.get("tick_size", 0.1))
    tp1_r = float(params.get("tp1_r", 1.0))
    tp1_frac = float(params.get("tp1_pct", 30.0)) / 100.0
    tp2_r = float(params.get("tp2_r", 2.0))
    min_gain = float(params.get("min_gain_pct", 0.5))
    # validade da ordem armada em candles; 999 = persiste até o viés virar
    # (melhor pareamento com o tester: 195/242 vs 182/242 com validade 2)
    armed_validity = int(params.get("armed_validity", 999))
    initial_capital = float(params.get("initial_capital", 1000.0))

    # Filtro de horário (intradiário): só entra nas horas permitidas (BRT).
    hour_filter = bool(params.get("hour_filter", False))
    allowed_hours = set(params.get("allowed_hours", []) or [])

    o = df["Open"].values
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    fast = df["Fast"].values
    slow = df["Slow"].values
    idx = df.index
    n = len(df)

    st = {
        "position": 0,          # 0 flat, 1 long, -1 short
        "entry": 0.0,
        "stop": 0.0,
        "tp1": 0.0,
        "tp2": 0.0,
        "size1": 0.0,           # fração da perna TP1 ainda aberta
        "size2": 0.0,           # fração da perna TP2 ainda aberta
        "exposure": 1.0,
        "leverage": 1.0,
        "qty": 0.0,
        "entry_bar": -1,
        "trade": None,
        "equity": initial_capital,
        "armed": False,
        "armed_dir": 0,
        "armed_level": 0.0,
        "armed_stop": 0.0,
        "armed_bar": -1,
    }
    trades = []
    equity_curve = []

    def _ts(i):
        try:
            return int(idx[i].value // 1_000_000)
        except AttributeError:
            return 0

    def _leg_exit(i, price, frac):
        """Realiza `frac` da posição a `price`; registra parcial no Trade."""
        d = st["position"]
        pnl = d * (price - st["entry"]) / st["entry"] * 100
        st["equity"] *= (1 + pnl / 100 * frac * st["exposure"])
        tr = st["trade"]
        # Trade inicializa partial_exit_price com 0.0 (não None)
        if not tr.partial_exit_price:
            tr.partial_exit_price = price
            tr.partial_exit_date = str(idx[i])
            tr.partial_pct_closed = frac
        return pnl

    def _close_all(i, price, comment):
        """Fecha o restante da posição e fecha o Trade."""
        tr = st["trade"]
        d = st["position"]
        remaining = st["size1"] + st["size2"]
        pnl_rest = d * (price - st["entry"]) / st["entry"] * 100
        st["equity"] *= (1 + pnl_rest / 100 * remaining * st["exposure"])

        tr.exit_date = str(idx[i])
        tr.exit_price = price
        tr.exit_comment = comment
        tr.exit_ts = _ts(i)
        # pnl total ponderado pelas frações já realizadas
        total = 0.0
        if tr.partial_exit_price and tr.partial_pct_closed:
            p_pnl = d * (tr.partial_exit_price - st["entry"]) / st["entry"] * 100
            total += tr.partial_pct_closed * p_pnl
        total += remaining * pnl_rest
        tr.pnl_pct = total * st["exposure"]
        trades.append(tr)

        st["position"] = 0
        st["entry"] = 0.0
        st["trade"] = None
        st["size1"] = st["size2"] = 0.0
        st["exposure"] = 1.0
        st["qty"] = 0.0
        st["entry_bar"] = -1

    def open_position(i, direction, entry_price, stop_price):
        risk = abs(entry_price - stop_price)
        if risk <= 0:
            return False
        qty, lev, exposure = _compute_sizing(st["equity"], entry_price,
                                             stop_price, params)
        st["position"] = direction
        st["entry"] = entry_price
        st["stop"] = stop_price
        st["tp1"] = entry_price + direction * tp1_r * risk
        st["tp2"] = entry_price + direction * tp2_r * risk
        st["size1"] = tp1_frac
        st["size2"] = 1.0 - tp1_frac
        st["exposure"] = exposure
        st["leverage"] = lev
        st["qty"] = qty
        st["entry_bar"] = i
        st["trade"] = Trade(
            entry_date=str(idx[i]),
            entry_price=entry_price,
            direction=direction,
            comment="L" if direction == 1 else "S",
            stop_price=stop_price,
            target_price=st["tp2"],
            qty=qty,
            leverage=lev,
            notional=qty * entry_price,
            exposure=exposure,
            entry_ts=_ts(i),
        )
        return True

    for i in range(n):
        if np.isnan(fast[i]) or np.isnan(slow[i]):
            equity_curve.append(st["equity"])
            continue

        bias = 1 if c[i] > slow[i] else (-1 if c[i] < slow[i] else 0)

        # ── 1. FILL da ordem armada (flat) — validade de `armed_validity`
        # candles após o gatilho (in_4=2 do TrapM; 93% dos fills em <=2) ────
        # Horário em Brasília (BRT), igual ao filtro da UI.
        _hour_ok = (not hour_filter) or (brt_hour(idx[i]) in allowed_hours)
        if st["position"] == 0 and st["armed"] \
                and i - st["armed_bar"] > armed_validity:
            st["armed"] = False
        if st["position"] == 0 and st["armed"] and _hour_ok:
            d = st["armed_dir"]
            if (d == 1 and h[i] > st["armed_level"]) or \
               (d == -1 and l[i] < st["armed_level"]):
                entry_price = (max(o[i], st["armed_level"]) if d == 1
                               else min(o[i], st["armed_level"]))
                open_position(i, d, entry_price, st["armed_stop"])
                st["armed"] = False

        # ── 2. SAÍDAS da posição (vale já no candle do fill) ────────────────
        if st["position"] != 0:
            d = st["position"]
            fill_bar = i == st["entry_bar"]
            # No candle do fill (entrada stop preenche a caminho do extremo),
            # o SL intrabar só é alcançável se o candle fechou CONTRA
            against = (c[i] < o[i]) if d == 1 else (c[i] > o[i])
            sl_reachable = (not fill_bar) or against

            # perna TP2: stop estrutural primeiro (conservador), depois alvo
            if st["size2"] > 0:
                hit_sl = ((l[i] <= st["stop"]) if d == 1 else (h[i] >= st["stop"]))
                if hit_sl and sl_reachable:
                    _leg_exit(i, st["stop"], st["size2"])
                    st["size2"] = 0.0
                else:
                    hit_tp2 = ((h[i] >= st["tp2"]) if d == 1 else (l[i] <= st["tp2"]))
                    if hit_tp2:
                        _leg_exit(i, st["tp2"], st["size2"])
                        st["size2"] = 0.0
            # perna TP1: só o alvo (sem stop próprio)
            if st["position"] != 0 and st["size1"] > 0:
                hit_tp1 = ((h[i] >= st["tp1"]) if d == 1 else (l[i] <= st["tp1"]))
                if hit_tp1:
                    _leg_exit(i, st["tp1"], st["size1"])
                    st["size1"] = 0.0
            # tudo realizado nos alvos/stop -> fecha o Trade com a última perna
            if st["position"] != 0 and st["size1"] + st["size2"] <= 0:
                tr = st["trade"]
                tr.exit_date = str(idx[i])
                tr.exit_price = st["tp2"] if tr.partial_exit_price == st["tp1"] else st["tp1"]
                tr.exit_comment = "Alvo 2" if tr.exit_price == st["tp2"] else "Alvo 1"
                tr.exit_ts = _ts(i)
                d0 = st["position"]
                p_pnl = d0 * (tr.partial_exit_price - st["entry"]) / st["entry"] * 100
                x_pnl = d0 * (tr.exit_price - st["entry"]) / st["entry"] * 100
                tr.pnl_pct = (tr.partial_pct_closed * p_pnl
                              + (1 - tr.partial_pct_closed) * x_pnl) * st["exposure"]
                trades.append(tr)
                st["position"] = 0
                st["trade"] = None
                st["size1"] = st["size2"] = 0.0
                st["exposure"] = 1.0
                st["entry_bar"] = -1
            # Slow MA Stop: close cruzou a lenta contra -> fecha TUDO no close
            elif st["position"] != 0:
                ma_stop = (c[i] < slow[i]) if d == 1 else (c[i] > slow[i])
                if ma_stop:
                    _close_all(i, c[i], "Slow MA Stop")

        # ── 3. NOVO GATILHO (substitui ordem armada; cancela se viés virou) ─
        if st["position"] == 0:
            if st["armed"] and ((st["armed_dir"] == 1 and bias < 0)
                                or (st["armed_dir"] == -1 and bias > 0)):
                st["armed"] = False
            if i >= 1 and bias != 0:
                d = bias
                # pullback: o candle de gatilho ABRE aquém da rápida (100%
                # dos gatilhos reais) e o candle ANTERIOR fecha aquém
                # (melhor equilíbrio precisão/recall — analyze_trapm4/5)
                prev_close_filter = bool(params.get("prev_close_filter", True))
                touches = ((o[i] <= fast[i] and
                            (not prev_close_filter or c[i - 1] <= fast[i - 1]))
                           if d == 1 else
                           (o[i] >= fast[i] and
                            (not prev_close_filter or c[i - 1] >= fast[i - 1])))
                pattern = ((use_engulfing and _is_engulfing(d, o[i], c[i], o[i - 1], c[i - 1]))
                           or (use_pfr and _is_pfr(d, h[i], l[i], h[i - 1], l[i - 1], c[i], c[i - 1])))
                if touches and pattern:
                    lo = max(0, i - stop_lookback + 1)
                    level = h[i] if d == 1 else l[i]
                    stop = (float(np.min(l[lo:i + 1])) - tick if d == 1
                            else float(np.max(h[lo:i + 1])) + tick)
                    risk_pct = abs(level - stop) / level * 100 if level > 0 else 0.0
                    if min_gain <= 0 or risk_pct >= min_gain:
                        st["armed"] = True
                        st["armed_dir"] = d
                        st["armed_level"] = level
                        st["armed_stop"] = stop
                        st["armed_bar"] = i

        # ── 4. Equity mark-to-market ────────────────────────────────────────
        if st["position"] != 0 and st["trade"] is not None:
            remaining = st["size1"] + st["size2"]
            unreal = st["position"] * (c[i] - st["entry"]) / st["entry"]
            equity_curve.append(st["equity"] * (1 + unreal * remaining * st["exposure"]))
        else:
            equity_curve.append(st["equity"])

    if st["position"] != 0 and n > 0:
        _close_all(n - 1, c[-1], "Fim do Período")
        if equity_curve:
            equity_curve[-1] = st["equity"]

    df["Equity"] = equity_curve[:n]
    return df, trades, st["equity"]


def run(df, params: dict) -> dict:
    """Executa o backtest e retorna metrics/equity_curve/trades/drawdown/chart."""
    initial_capital = float(params.get("initial_capital", 1000.0))
    bt_df, trades, final_equity = _run_backtest_mm9(df, params)

    total_return = (final_equity / initial_capital - 1) * 100
    wins = [t for t in trades if t.pnl_pct > 0]
    losses_t = [t for t in trades if t.pnl_pct <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0
    avg_loss = float(np.mean([t.pnl_pct for t in losses_t])) if losses_t else 0
    pf_num = sum(t.pnl_pct for t in wins)
    pf_den = abs(sum(t.pnl_pct for t in losses_t))
    profit_factor = pf_num / pf_den if pf_den > 0 else None

    eq = bt_df["Equity"].values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = float(dd.min()) if len(dd) else 0.0

    total_days = max((bt_df.index[-1] - bt_df.index[0]).days, 1) if len(bt_df) > 1 else 1
    cagr = ((final_equity / initial_capital) ** (365.25 / total_days) - 1) * 100
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 0 else 0

    bars_per_year = len(bt_df) / (total_days / 365.25) if total_days > 0 else 252

    sharpe = 0.0
    pnls = [t.pnl_pct for t in trades]
    if len(eq) > 2:
        eq_rets = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1.0)
        eq_rets = eq_rets[np.isfinite(eq_rets)]
        eq_std = float(eq_rets.std(ddof=1)) if len(eq_rets) > 1 else 0.0
        if len(eq_rets) > 1 and eq_std > 0:
            sharpe = float(eq_rets.mean() / eq_std * np.sqrt(bars_per_year))

    downside_sq = [min(p, 0) ** 2 for p in pnls]
    downside_dev = float(np.sqrt(np.mean(downside_sq))) if downside_sq else 0
    trades_per_year = len(pnls) / (total_days / 365.25) if total_days > 0 else len(pnls)
    sortino = float(np.mean(pnls) / downside_dev * np.sqrt(trades_per_year)) if downside_dev > 0 else 0

    gains_sum = sum(p for p in pnls if p > 0)
    losses_sum = abs(sum(p for p in pnls if p < 0))
    omega = float(gains_sum / losses_sum) if losses_sum > 0 else None

    ep_troughs = []
    _in_dd = False
    _ep_start = None
    for _k, _val in enumerate(dd):
        if _val < 0 and not _in_dd:
            _in_dd = True
            _ep_start = _k
        elif _val >= 0 and _in_dd:
            _in_dd = False
            ep_troughs.append(float(dd[_ep_start:_k].min()))
    if _in_dd and _ep_start is not None:
        ep_troughs.append(float(dd[_ep_start:].min()))

    if ep_troughs:
        n_worst = min(5, len(ep_troughs))
        worst_troughs = sorted(ep_troughs)[:n_worst]
        avg_worst_dd = abs(float(np.mean(worst_troughs)))
        sterling = float(cagr / avg_worst_dd) if avg_worst_dd > 0 else 0
        burke_denom = float(np.sqrt(np.sum(np.array(worst_troughs) ** 2)))
        burke = float(cagr / burke_denom) if burke_denom > 0 else 0
    else:
        sterling = 0
        burke = 0

    _fast = bool(params.get("_fast", False))
    dates = [str(idx)[:10] for idx in bt_df.index]

    if _fast:
        recovery_factor = None
        ulcer_index = None
        avg_dd = None
        avg_dd_len = None
        n_dd_episodes = 0
        dd_dates = []
        dd_values = []
        dd_episodes = []
        equity_values = [_safe(float(v)) for v in bt_df["Equity"].tolist()]
        trades_list = [
            {
                "pnl_pct": _safe(float(t.pnl_pct)),
                "qty": _safe(float(t.qty)),
                "leverage": _safe(float(t.leverage)),
                "notional": _safe(float(t.notional)),
                "direction": t.direction,
                "entry_price": _safe(float(t.entry_price)),
                "exit_price": _safe(float(t.exit_price)),
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
            }
            for t in trades
        ]
    else:
        net_profit = final_equity - initial_capital
        recovery_factor = _safe(float(net_profit / abs(max_dd * initial_capital / 100))) if max_dd < 0 else None
        ulcer_index = _safe(float(np.sqrt(np.mean(dd ** 2)))) if len(dd) > 0 else None

        dd_episodes = []
        in_dd = False
        ep_start = None
        for k, val in enumerate(dd):
            if val < 0 and not in_dd:
                in_dd = True
                ep_start = k
            elif val >= 0 and in_dd:
                in_dd = False
                dd_episodes.append({
                    "start": dates[ep_start] if ep_start < len(bt_df) else None,
                    "end": dates[k - 1] if (k - 1) < len(bt_df) else None,
                    "trough": _safe(float(dd[ep_start:k].min())),
                    "length_bars": k - ep_start,
                })
        if in_dd and ep_start is not None:
            dd_episodes.append({
                "start": dates[ep_start],
                "end": dates[-1],
                "trough": _safe(float(dd[ep_start:].min())),
                "length_bars": len(dd) - ep_start,
            })

        avg_dd = _safe(float(np.mean([e["trough"] for e in dd_episodes]))) if dd_episodes else None
        avg_dd_len = _safe(float(np.mean([e["length_bars"] for e in dd_episodes]))) if dd_episodes else None
        n_dd_episodes = len(dd_episodes)
        dd_dates = dates
        dd_values = [_safe(float(v)) for v in dd.tolist()]
        equity_values = [_safe(float(v)) for v in bt_df["Equity"].tolist()]

        trades_list = [
            {
                "entry_date": t.entry_date[:10] if t.entry_date else "",
                "exit_date": t.exit_date[:10] if t.exit_date else "",
                "direction": t.direction,
                "comment": t.comment,
                "entry_price": _safe(float(t.entry_price)),
                "exit_price": _safe(float(t.exit_price)),
                "exit_comment": t.exit_comment,
                "stop_price": _safe(float(t.stop_price)) if t.stop_price else None,
                "target_price": _safe(float(t.target_price)) if t.target_price else None,
                "pnl_pct": _safe(float(t.pnl_pct)),
                "partial_exit_price": _safe(float(t.partial_exit_price)) if t.partial_exit_price else None,
                "partial_exit_date": t.partial_exit_date[:10] if t.partial_exit_date else None,
                "partial_pct_closed": t.partial_pct_closed if t.partial_pct_closed else None,
                "qty": _safe(float(t.qty)),
                "leverage": _safe(float(t.leverage)),
                "exposure": _safe(float(t.exposure)),
                "notional": _safe(float(t.notional)),
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
            }
            for t in trades
        ]

    chart = None
    if params.get("_charts"):
        def _col(name):
            return [_safe(float(v)) for v in bt_df[name].tolist()] if name in bt_df.columns else None
        chart = {
            "dates": [str(idx) for idx in bt_df.index],
            "ohlc": {
                "open": _col("Open"),
                "high": _col("High"),
                "low": _col("Low"),
                "close": _col("Close"),
                "volume": _col("Volume"),
            },
            "indicators": {
                "ma": _col("Fast"),
                "ma_slow": _col("Slow"),
            },
        }

    return {
        "chart": chart,
        "metrics": {
            "final_equity": _safe(float(final_equity)),
            "total_return": _safe(float(total_return)),
            "max_dd": _safe(max_dd),
            "total_trades": len(trades),
            "win_rate": _safe(float(win_rate)),
            "profit_factor": _safe(profit_factor),
            "avg_win": _safe(avg_win),
            "avg_loss": _safe(avg_loss),
            "sharpe": _safe(sharpe),
            "sortino": _safe(sortino),
            "calmar": _safe(calmar),
            "omega": _safe(omega),
            "sterling": _safe(sterling),
            "burke": _safe(burke),
            "recovery_factor": recovery_factor,
            "ulcer_index": ulcer_index,
            "avg_dd": avg_dd,
            "avg_dd_length": avg_dd_len,
            "n_dd_episodes": n_dd_episodes,
            "initial_capital": float(initial_capital),
        },
        "drawdown": {
            "dates": dd_dates,
            "values": dd_values,
            "episodes": dd_episodes if not _fast else [],
        },
        "equity_curve": {
            "dates": dates,
            "values": equity_values,
        },
        "trades": trades_list,
    }
