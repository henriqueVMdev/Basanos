"""
Estratégia DePaula Single + Pullback [v2] — Backtesting em Python
Conversão fiel do Pine Script para Python usando pandas/numpy.

Uso:
    python -m engine.backtesting                          # usa dados de exemplo via yfinance
    python -m engine.backtesting --csv dados.csv          # usa CSV local (colunas: Date,Open,High,Low,Close,Volume)
    python -m engine.backtesting --symbol BTCUSDT --interval 1d
"""

import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field
from typing import Optional
import argparse
import sys
from pathlib import Path


# 1. CONFIGURAÇÃO (equivale aos inputs do Pine Script)
@dataclass
class Config:
    # Média Principal
    ma_type: str = "HMA"           # SMA, EMA, HMA, RMA, WMA
    ma_length: int = 50
    lookback: int = 5
    th_up: float = 0.5
    th_dn: float = -0.5

    # Modo de Saída
    exit_mode: str = "Banda + Tendência"  # "Banda + Tendência", "Somente Tendência", "Alvo Fixo + Tendência"
    pct_up: float = 3.0
    pct_dn: float = 3.0
    alvo_fixo: float = 5.0
    exit_on_flat: bool = True

    # Stop Loss
    use_stop: bool = False
    stop_type: str = "ATR"         # "ATR", "Fixo (%)", "Banda Stop"
    stop_atr_mult: float = 2.0
    stop_fixo_pct: float = 2.0
    stop_band_pct: float = 1.5

    # Entrada
    use_pullback: bool = True
    use_entry_zone: bool = False

    # Global
    use_norm: bool = True
    atr_length: int = 14
    hysteresis: float = 0.2

    # Saida parcial
    use_parcial: bool = False
    parcial_pct: float = 50.0           # % da posicao a realizar
    parcial_mode: str = "Banda"         # "Banda" ou "Alvo Fixo"
    parcial_banda_pct: float = 1.5      # % da MA para banda parcial
    parcial_alvo_fixo: float = 2.0      # % fixo do preco de entrada

    # Position sizing
    use_position_sizing: bool = False
    risk_per_trade: float = 1.0  # % do capital arriscado por trade

    # Sizing / alavancagem (futuros)
    # sizing_mode: "Alavancagem fixa" | "Quantidade fixa" | "Risco por trade"
    sizing_mode: str = "Alavancagem fixa"
    leverage: float = 1.0          # alavancagem (x). 1.0 = sem alavancagem
    margin_pct: float = 100.0      # % do equity usado como margem por trade (modo Alavancagem fixa)
    fixed_qty: float = 0.1         # quantidade fixa em unidades do ativo (modo Quantidade fixa)

    # Backtest
    initial_capital: float = 1000.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Ciclo sazonal
    cycle_filter: bool = False
    cycle_long_months: list = field(default_factory=list)   # ex: [8,9,10,11,12,1,2]
    cycle_short_months: list = field(default_factory=list)  # ex: [3,4,5,6,7]

    # Filtro de horário (intradiário): só abre posição nas horas permitidas (BRT).
    hour_filter: bool = False
    allowed_hours: list = field(default_factory=list)       # ex: [9,10,11,14,15] (0-23 BRT)


# 2. INDICADORES (equivale às funções do Pine Script)
def calc_sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()


def calc_ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def calc_rma(series: pd.Series, length: int) -> pd.Series:
    """RMA = Wilder's smoothing (equivalente a EMA com alpha=1/length)"""
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


def calc_wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(window=length, min_periods=length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def calc_hma(series: pd.Series, length: int) -> pd.Series:
    half_len = max(int(length / 2), 1)
    sqrt_len = max(int(math.sqrt(length)), 1)
    wma_half = calc_wma(series, half_len)
    wma_full = calc_wma(series, length)
    diff = 2 * wma_half - wma_full
    return calc_wma(diff, sqrt_len)


def calc_ma(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    funcs = {
        "SMA": calc_sma,
        "EMA": calc_ema,
        "HMA": calc_hma,
        "RMA": calc_rma,
        "WMA": calc_wma,
    }
    return funcs.get(ma_type, calc_sma)(series, length)


def calc_atr(df: pd.DataFrame, length: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return calc_rma(tr, length)


def calc_slope(ma_values: np.ndarray, idx: int, length: int) -> float:
    """Regressão linear sobre os últimos 'length' valores da MA."""
    safe_len = max(length, 2)
    if idx < safe_len - 1:
        return 0.0

    sum_x = sum_y = sum_xy = sum_xx = 0.0
    for i in range(safe_len):
        x = -i
        y = ma_values[idx - i]
        if np.isnan(y):
            return 0.0
        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_xx += x * x

    denom = safe_len * sum_xx - sum_x * sum_x
    if denom == 0.0:
        return 0.0
    return (safe_len * sum_xy - sum_x * sum_y) / denom


def calc_angle(slope: float, atr_val: float, use_norm: bool, mintick: float = 0.01) -> float:
    if use_norm:
        norm = slope / max(atr_val, mintick) if atr_val > 0 else slope
    else:
        norm = slope
    return math.degrees(math.atan(norm))


# 3. PREPARAÇÃO DOS DADOS
def prepare_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Calcula todos os indicadores e adiciona ao DataFrame."""
    df = df.copy()

    # Média Móvel
    df["MA"] = calc_ma(df["Close"], cfg.ma_length, cfg.ma_type)

    # ATR
    df["ATR"] = calc_atr(df, cfg.atr_length)

    # Bandas de saída
    df["UpperBand"] = df["MA"] * (1 + cfg.pct_up / 100)
    df["LowerBand"] = df["MA"] * (1 - cfg.pct_dn / 100)

    # Bandas de stop
    df["StopUpperBand"] = df["MA"] * (1 + cfg.stop_band_pct / 100)
    df["StopLowerBand"] = df["MA"] * (1 - cfg.stop_band_pct / 100)

    # Bandas de saida parcial
    df["PartialUpperBand"] = df["MA"] * (1 + cfg.parcial_banda_pct / 100)
    df["PartialLowerBand"] = df["MA"] * (1 - cfg.parcial_banda_pct / 100)

    # Slope e Angle (barra a barra)
    ma_vals = df["MA"].values
    atr_vals = df["ATR"].values
    angles = np.zeros(len(df))

    for i in range(len(df)):
        if np.isnan(ma_vals[i]) or np.isnan(atr_vals[i]):
            angles[i] = 0.0
            continue
        slope = calc_slope(ma_vals, i, cfg.lookback)
        angles[i] = calc_angle(slope, atr_vals[i], cfg.use_norm)

    df["Angle"] = angles

    # Máquina de Estado (barra a barra com histerese)
    states = np.zeros(len(df), dtype=int)
    state = 0

    for i in range(len(df)):
        angle = angles[i]
        if state <= 0 and angle > cfg.th_up + cfg.hysteresis:
            state = 1
        elif state >= 0 and angle < cfg.th_dn - cfg.hysteresis:
            state = -1
        elif state == 1 and angle < cfg.th_up - cfg.hysteresis:
            state = 0
        elif state == -1 and angle > cfg.th_dn + cfg.hysteresis:
            state = 0
        states[i] = state

    df["State"] = states
    df["PrevState"] = pd.Series(states, index=df.index).shift(1).fillna(0).astype(int)

    # Zona de entrada
    df["InEntryZone"] = (df["Close"] >= df["StopLowerBand"]) & (df["Close"] <= df["StopUpperBand"])

    return df


# 4. ENGINE DE BACKTESTING
@dataclass
class Trade:
    entry_date: str
    entry_price: float
    direction: int          # 1 = Long, -1 = Short
    comment: str
    exit_date: str = ""
    exit_price: float = 0.0
    exit_comment: str = ""
    pnl_pct: float = 0.0
    partial_exit_price: float = 0.0
    partial_exit_date: str = ""
    partial_pct_closed: float = 0.0
    # Niveis da operacao (para visualizacao no grafico). 0.0 = nao definido.
    stop_price: float = 0.0
    target_price: float = 0.0
    # Sizing / custos (futuros)
    qty: float = 0.0            # quantidade em unidades do ativo
    leverage: float = 1.0       # alavancagem aplicada (x)
    notional: float = 0.0       # qty * entry_price (notional alavancado)
    exposure: float = 1.0       # notional / equity_no_momento_da_entrada
    entry_ts: int = 0           # epoch ms da entrada
    exit_ts: int = 0            # epoch ms da saida


@dataclass
class BacktestState:
    position: int = 0           # 0=flat, 1=long, -1=short
    entry_price: float = 0.0
    aguardando_pullback: bool = False
    pending_long: bool = False
    pending_short: bool = False
    equity: float = 1000.0
    trades: list = field(default_factory=list)
    current_trade: Optional[Trade] = None
    partial_taken: bool = False
    position_size: float = 1.0  # 1.0 = full, reduz apos parcial
    position_fraction: float = 1.0  # exposure = notional / equity (inclui alavancagem)
    leverage: float = 1.0           # alavancagem do trade aberto (p/ limite de liquidacao)
    qty: float = 0.0                # quantidade do trade aberto (unidades)


def _cycle_allows(cfg: Config, month: int, direction: int) -> bool:
    """Checa se o ciclo sazonal permite abrir posicao nesse mes/direcao."""
    if not cfg.cycle_filter:
        return True
    if direction == 1:
        return month in cfg.cycle_long_months if cfg.cycle_long_months else True
    if direction == -1:
        return month in cfg.cycle_short_months if cfg.cycle_short_months else True
    return True


def _hour_allows(cfg: Config, hour: int) -> bool:
    """Checa se o filtro de horário permite abrir posição nessa hora (0-23)."""
    if not cfg.hour_filter:
        return True
    # Lista vazia = nenhuma hora liberada (não opera).
    return hour in cfg.allowed_hours


def brt_hour(ts) -> int:
    """Hora do timestamp em horário de Brasília (America/Sao_Paulo).

    Barras tz-aware são convertidas; barras naive são assumidas em UTC (UTC-3).
    """
    if not hasattr(ts, "hour"):
        return 0
    try:
        if getattr(ts, "tzinfo", None) is not None:
            return ts.tz_convert("America/Sao_Paulo").hour
    except Exception:
        pass
    return (ts.hour - 3) % 24


def _bar_index(idx):
    """Pré-calcula (datas, meses, horas BRT, epoch ms) do índice das barras."""
    dates = [str(x) for x in idx]
    if not isinstance(idx, pd.DatetimeIndex):
        zeros = [0] * len(idx)
        return dates, [1] * len(idx), zeros, zeros
    local = idx.tz_convert("America/Sao_Paulo") if idx.tz is not None \
        else idx - pd.Timedelta(hours=3)
    # as_unit("ns"): o índice pode ser datetime64[us/ms] e asi8 segue a unidade
    ts_ms = (idx.as_unit("ns").asi8 // 1_000_000).tolist()
    return dates, idx.month.tolist(), local.hour.tolist(), ts_ms


def run_backtest(df: pd.DataFrame, cfg: Config) -> BacktestState:
    """Executa o backtest barra a barra, replicando a lógica do Pine Script."""

    df = prepare_indicators(df, cfg)
    st = BacktestState(equity=cfg.initial_capital)

    use_bands = cfg.exit_mode == "Banda + Tendência"
    use_alvo_fixo = cfg.exit_mode == "Alvo Fixo + Tendência"

    # Filtro de data
    if cfg.start_date:
        df = df[df.index >= cfg.start_date]
    if cfg.end_date:
        df = df[df.index <= cfg.end_date]

    equity_curve = []

    # df.iloc[i] por barra domina o custo do loop (o optimizer roda isto
    # milhares de vezes): extrai as colunas uma vez e itera sobre arrays.
    col = {name: df[name].to_numpy() for name in
           ("Close", "High", "Low", "MA", "ATR", "UpperBand", "LowerBand",
            "StopUpperBand", "StopLowerBand", "PartialUpperBand",
            "PartialLowerBand", "State", "PrevState", "InEntryZone")}
    dates, months, hours, ts_ms = _bar_index(df.index)

    for i in range(len(df)):
        date = dates[i]
        bar_month = months[i]
        bar_hour = hours[i]          # hora de Brasília (UI usa horário BRT)
        st._ts = ts_ms[i]            # epoch ms da barra (usado em custos/funding)
        close = col["Close"][i]
        high = col["High"][i]
        low = col["Low"][i]
        state = int(col["State"][i])
        prev_state = int(col["PrevState"][i])
        ma_val = col["MA"][i]
        atr_val = col["ATR"][i]
        upper_band = col["UpperBand"][i]
        lower_band = col["LowerBand"][i]
        stop_upper = col["StopUpperBand"][i]
        stop_lower = col["StopLowerBand"][i]
        partial_upper = col["PartialUpperBand"][i]
        partial_lower = col["PartialLowerBand"][i]
        in_entry_zone = bool(col["InEntryZone"][i])

        if np.isnan(ma_val) or np.isnan(atr_val):
            equity_curve.append(st.equity)
            continue

        # MUDANÇA DE ESTADO: reset pullback e marca pendentes
        if state != prev_state:
            st.aguardando_pullback = False
            st.pending_long = (state == 1)
            st.pending_short = (state == -1)

        # SAIDA PARCIAL (antes do TP/SL)
        if (cfg.use_parcial and st.position != 0
                and not st.partial_taken and st.current_trade is not None):
            parcial_price = None
            if cfg.parcial_mode == "Banda":
                parcial_price = partial_upper if st.position == 1 else partial_lower
            elif cfg.parcial_mode == "Alvo Fixo":
                if st.position == 1:
                    parcial_price = st.entry_price * (1 + cfg.parcial_alvo_fixo / 100)
                else:
                    parcial_price = st.entry_price * (1 - cfg.parcial_alvo_fixo / 100)

            if parcial_price is not None:
                hit_parcial = False
                if st.position == 1 and high >= parcial_price:
                    hit_parcial = True
                elif st.position == -1 and low <= parcial_price:
                    hit_parcial = True

                if hit_parcial:
                    fraction = cfg.parcial_pct / 100.0
                    partial_pnl = st.position * (parcial_price - st.entry_price) / st.entry_price * 100
                    st.equity *= (1 + partial_pnl / 100 * fraction * st.position_fraction)
                    st.position_size -= fraction
                    st.partial_taken = True
                    st.current_trade.partial_exit_price = parcial_price
                    st.current_trade.partial_exit_date = date
                    st.current_trade.partial_pct_closed = fraction

        # SAÍDAS INTRABAR (TP + SL checados contra High/Low)
        if st.position != 0 and st.current_trade is not None:
            tp_price = None
            sl_price = None

            # TP
            if use_bands:
                tp_price = upper_band if st.position == 1 else lower_band
            elif use_alvo_fixo:
                if st.position == 1:
                    tp_price = st.entry_price * (1 + cfg.alvo_fixo / 100)
                else:
                    tp_price = st.entry_price * (1 - cfg.alvo_fixo / 100)

            # SL
            if cfg.use_stop:
                if cfg.stop_type == "ATR":
                    if st.position == 1:
                        sl_price = st.entry_price - cfg.stop_atr_mult * atr_val
                    else:
                        sl_price = st.entry_price + cfg.stop_atr_mult * atr_val
                elif cfg.stop_type == "Fixo (%)":
                    if st.position == 1:
                        sl_price = st.entry_price * (1 - cfg.stop_fixo_pct / 100)
                    else:
                        sl_price = st.entry_price * (1 + cfg.stop_fixo_pct / 100)
                elif cfg.stop_type == "Banda Stop":
                    sl_price = stop_lower if st.position == 1 else stop_upper

            # Preço de FALÊNCIA: perda adversa >= 1/alavancagem zera a margem.
            # Sem margem de manutenção nem fees: a liquidação real de uma exchange
            # ocorre um pouco ANTES disso, então esta aproximação é levemente otimista.
            liq_price = None
            if st.leverage and st.leverage > 1.0:
                if st.position == 1:
                    liq_price = st.entry_price * (1 - 1.0 / st.leverage)
                else:
                    liq_price = st.entry_price * (1 + 1.0 / st.leverage)

            # Se o stop está mais distante que a liquidação, a liquidação ocorre antes.
            # Trata a liquidação como um "stop" no liq_price quando for o gatilho adverso mais próximo.
            sl_is_liq = False
            if liq_price is not None:
                if sl_price is None:
                    sl_price = liq_price
                    sl_is_liq = True
                elif st.position == 1 and liq_price > sl_price:
                    sl_price = liq_price
                    sl_is_liq = True
                elif st.position == -1 and liq_price < sl_price:
                    sl_price = liq_price
                    sl_is_liq = True

            # Checa SL/liquidação primeiro (prioridade ao stop)
            hit_sl = False
            hit_tp = False

            if sl_price is not None:
                if st.position == 1 and low <= sl_price:
                    hit_sl = True
                elif st.position == -1 and high >= sl_price:
                    hit_sl = True

            if tp_price is not None and not hit_sl:
                if st.position == 1 and high >= tp_price:
                    hit_tp = True
                elif st.position == -1 and low <= tp_price:
                    hit_tp = True

            if hit_sl:
                _close_position(st, date, sl_price, "Liquidação" if sl_is_liq else "Stop Loss")
                if cfg.use_pullback:
                    if state == 1 or state == -1:
                        st.aguardando_pullback = True
            elif hit_tp:
                _close_position(st, date, tp_price, "Alvo Atingido")
                if cfg.use_pullback:
                    if state == 1 or state == -1:
                        st.aguardando_pullback = True

        # FIM DE TENDÊNCIA
        if st.position != 0:
            if cfg.exit_on_flat and state == 0:
                _close_position(st, date, close, "Fim Tendência (Cinza)")
            elif not cfg.exit_on_flat:
                if prev_state == 1 and state == -1:
                    _close_position(st, date, close, "Inversão L→S")
                elif prev_state == -1 and state == 1:
                    _close_position(st, date, close, "Inversão S→L")

        # DETECTA SAÍDA POR ALVO (pullback)
        # (já tratado acima no hit_sl/hit_tp)

        # ENTRADAS
        # A. Entrada de Tendência
        if not st.aguardando_pullback and (not cfg.use_entry_zone or in_entry_zone) and _hour_allows(cfg, bar_hour):
            if st.pending_long and state == 1 and st.position <= 0 and _cycle_allows(cfg, bar_month, 1):
                if st.position == -1:
                    _close_position(st, date, close, "Reversão S→L")
                _open_position(st, date, close, 1, "L Trend", cfg, atr_val)
                st.pending_long = False
            elif st.pending_short and state == -1 and st.position >= 0 and _cycle_allows(cfg, bar_month, -1):
                if st.position == 1:
                    _close_position(st, date, close, "Reversão L→S")
                _open_position(st, date, close, -1, "S Trend", cfg, atr_val)
                st.pending_short = False

        # B. Reentrada por Pullback
        if cfg.use_pullback and st.aguardando_pullback and st.position == 0 and _hour_allows(cfg, bar_hour):
            if cfg.use_entry_zone:
                pb_long = in_entry_zone
                pb_short = in_entry_zone
            else:
                pb_long = close < ma_val
                pb_short = close > ma_val

            if state == 1 and pb_long and _cycle_allows(cfg, bar_month, 1):
                _open_position(st, date, close, 1, "L Pullback", cfg, atr_val)
                st.aguardando_pullback = False
            elif state == -1 and pb_short and _cycle_allows(cfg, bar_month, -1):
                _open_position(st, date, close, -1, "S Pullback", cfg, atr_val)
                st.aguardando_pullback = False

        # Equity mark-to-market
        if st.position != 0 and st.current_trade is not None:
            unrealized = st.position * (close - st.entry_price) / st.entry_price
            equity_curve.append(st.equity * (1 + unrealized * st.position_size * st.position_fraction))
        else:
            equity_curve.append(st.equity)

    # Fecha posição aberta no final
    if st.position != 0 and len(df) > 0:
        _close_position(st, dates[-1], col["Close"][-1], "Fim do Período")

    df["Equity"] = equity_curve[:len(df)]
    st._df = df
    return st


def _calc_stop_distance_pct(cfg: Config, price: float, atr_val: float) -> float:
    """Calcula a distancia do stop em % do preco de entrada."""
    if not cfg.use_stop:
        return 0.0
    if cfg.stop_type == "ATR":
        return (cfg.stop_atr_mult * atr_val / price) * 100 if price > 0 else 0.0
    elif cfg.stop_type == "Fixo (%)":
        return cfg.stop_fixo_pct
    elif cfg.stop_type == "Banda Stop":
        return cfg.stop_band_pct
    return 0.0


def _open_position(st: BacktestState, date: str, price: float, direction: int, comment: str,
                   cfg: Config = None, atr_val: float = 0.0):
    st.position = direction
    st.entry_price = price
    st.partial_taken = False
    st.position_size = 1.0

    equity = st.equity if st.equity > 0 else 1.0
    leverage = 1.0
    exposure = 1.0          # notional / equity
    qty = 0.0

    if cfg is not None:
        mode = getattr(cfg, "sizing_mode", "Alavancagem fixa")
        lev = max(float(getattr(cfg, "leverage", 1.0)), 1.0)

        # Compatibilidade: flag antiga força o modo de risco
        if cfg.use_position_sizing and cfg.use_stop:
            mode = "Risco por trade"

        if mode == "Quantidade fixa":
            qty = max(float(getattr(cfg, "fixed_qty", 0.0)), 0.0)
            exposure = (qty * price) / equity
            leverage = lev
        elif mode == "Risco por trade":
            stop_dist = _calc_stop_distance_pct(cfg, price, atr_val)  # % do preço
            if stop_dist > 0:
                # notional tal que a perda no stop = risk_per_trade% do equity
                exposure = (cfg.risk_per_trade / 100.0) / (stop_dist / 100.0)
            else:
                exposure = 1.0
            leverage = lev
            if exposure > lev:          # não excede a alavancagem máxima permitida
                exposure = lev
            qty = (exposure * equity) / price if price > 0 else 0.0
        else:  # "Alavancagem fixa"
            margin_frac = max(min(float(getattr(cfg, "margin_pct", 100.0)) / 100.0, 1.0), 0.0)
            leverage = lev
            exposure = margin_frac * lev
            qty = (exposure * equity) / price if price > 0 else 0.0

    st.position_fraction = exposure
    st.leverage = leverage
    st.qty = qty

    st.current_trade = Trade(
        entry_date=date,
        entry_price=price,
        direction=direction,
        comment=comment,
        qty=qty,
        leverage=leverage,
        notional=qty * price,
        exposure=exposure,
        entry_ts=getattr(st, "_ts", 0),
    )


def _close_position(st: BacktestState, date: str, price: float, comment: str):
    if st.current_trade is None:
        st.position = 0
        return

    trade = st.current_trade
    trade.exit_date = date
    trade.exit_price = price
    trade.exit_comment = comment
    trade.exit_ts = getattr(st, "_ts", 0)

    frac = st.position_fraction
    remaining_pnl = trade.direction * (price - trade.entry_price) / trade.entry_price * 100

    # Se houve saida parcial, aplica apenas o PnL da parte restante
    st.equity *= (1 + remaining_pnl / 100 * st.position_size * frac)

    # PnL total combinado (parcial + restante), ja ponderado pelo sizing
    if st.partial_taken and trade.partial_pct_closed > 0:
        partial_pnl = trade.direction * (trade.partial_exit_price - trade.entry_price) / trade.entry_price * 100
        trade.pnl_pct = (trade.partial_pct_closed * partial_pnl
                         + st.position_size * remaining_pnl) * frac
    else:
        trade.pnl_pct = remaining_pnl * frac

    st.trades.append(trade)
    st.position = 0
    st.entry_price = 0.0
    st.current_trade = None
    st.partial_taken = False
    st.position_size = 1.0
    st.position_fraction = 1.0
    st.leverage = 1.0
    st.qty = 0.0


# 5. RELATÓRIO
def print_report(st: BacktestState, cfg: Config):
    """Imprime relatório completo do backtest."""

    trades = st.trades
    if not trades:
        print("\n  Nenhum trade executado.")
        return

    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    pnls = [t.pnl_pct for t in trades]

    total_return = (st.equity / cfg.initial_capital - 1) * 100
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    profit_factor = abs(sum(t.pnl_pct for t in wins) / sum(t.pnl_pct for t in losses)) if losses and sum(t.pnl_pct for t in losses) != 0 else float("inf")

    # Max drawdown da equity curve
    df = st._df
    if "Equity" in df.columns:
        eq = df["Equity"].values
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        max_dd = dd.min()
    else:
        max_dd = 0

    print("\n" + "=" * 60)
    print("  RELATÓRIO DE BACKTEST — Estratégia DePaula v2")
    print("=" * 60)

    print(f"\n  Configuração:")
    print(f"    Média: {cfg.ma_type}({cfg.ma_length})  |  Lookback: {cfg.lookback}")
    print(f"    Ângulo Alta: {cfg.th_up}°  |  Baixa: {cfg.th_dn}°  |  Histerese: {cfg.hysteresis}°")
    print(f"    Saída: {cfg.exit_mode}  |  Flat: {'SIM' if cfg.exit_on_flat else 'NÃO'}")
    print(f"    Stop: {'OFF' if not cfg.use_stop else cfg.stop_type}")
    print(f"    Pullback: {'ON' if cfg.use_pullback else 'OFF'}  |  Entry Zone: {'ON' if cfg.use_entry_zone else 'OFF'}")

    print(f"\n  Resultados:")
    print(f"    Capital Inicial:   ${cfg.initial_capital:,.2f}")
    print(f"    Capital Final:     ${st.equity:,.2f}")
    print(f"    Retorno Total:     {total_return:+.2f}%")
    print(f"    Max Drawdown:      {max_dd:.2f}%")

    print(f"\n  Trades:")
    print(f"    Total:             {len(trades)}")
    print(f"    Vencedores:        {len(wins)} ({win_rate:.1f}%)")
    print(f"    Perdedores:        {len(losses)} ({100 - win_rate:.1f}%)")
    print(f"    Média Ganho:       {avg_win:+.2f}%")
    print(f"    Média Perda:       {avg_loss:+.2f}%")
    print(f"    Profit Factor:     {profit_factor:.2f}")
    print(f"    Maior Ganho:       {max(pnls):+.2f}%")
    print(f"    Maior Perda:       {min(pnls):+.2f}%")

    print(f"\n  Últimos 10 trades:")
    print(f"    {'Data Entrada':<22} {'Tipo':<12} {'Entrada':>10} {'Saída':>10} {'P&L':>8}  {'Motivo Saída'}")
    print(f"    {'-'*22} {'-'*12} {'-'*10} {'-'*10} {'-'*8}  {'-'*20}")
    for t in trades[-10:]:
        dir_str = "LONG" if t.direction == 1 else "SHORT"
        date_str = t.entry_date[:10] if len(t.entry_date) > 10 else t.entry_date
        print(f"    {date_str:<22} {dir_str + ' ' + t.comment:<12} {t.entry_price:>10.2f} {t.exit_price:>10.2f} {t.pnl_pct:>+7.2f}%  {t.exit_comment}")

    print("\n" + "=" * 60)


# 6. DADOS
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV precisa das colunas: {required}. Encontradas: {set(df.columns)}")
    return df.sort_index()


def download_data(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 1000) -> pd.DataFrame:
    """Tenta baixar dados via yfinance."""
    try:
        import yfinance as yf
        # Mapeia símbolos cripto para yfinance
        yf_symbol = symbol
        if "USDT" in symbol.upper():
            yf_symbol = symbol.upper().replace("USDT", "-USD")

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period="max", interval=interval)
        if df.empty:
            raise ValueError(f"Nenhum dado retornado para {yf_symbol}")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        print(f"  Dados carregados: {yf_symbol} | {len(df)} barras | {df.index[0].date()} a {df.index[-1].date()}")
        return df
    except ImportError:
        print("  ERRO: yfinance não instalado. Instale com: pip install yfinance")
        print("  Ou use --csv para carregar dados de um arquivo CSV.")
        sys.exit(1)


# 7. MAIN
def main():
    parser = argparse.ArgumentParser(description="Backtesting — Estratégia DePaula v2")

    # Dados
    parser.add_argument("--csv", type=str, help="Caminho do CSV (Date,Open,High,Low,Close)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Símbolo (default: BTCUSDT)")
    parser.add_argument("--interval", type=str, default="1d", help="Intervalo (default: 1d)")

    # Média
    parser.add_argument("--ma-type", type=str, default="HMA", choices=["SMA", "EMA", "HMA", "RMA", "WMA"])
    parser.add_argument("--ma-length", type=int, default=50)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--th-up", type=float, default=0.5)
    parser.add_argument("--th-dn", type=float, default=-0.5)
    parser.add_argument("--hysteresis", type=float, default=0.2)

    # Saída
    parser.add_argument("--exit-mode", type=str, default="Banda + Tendência",
                        choices=["Banda + Tendência", "Somente Tendência", "Alvo Fixo + Tendência"])
    parser.add_argument("--pct-up", type=float, default=3.0)
    parser.add_argument("--pct-dn", type=float, default=3.0)
    parser.add_argument("--alvo-fixo", type=float, default=5.0)
    parser.add_argument("--exit-on-flat", action="store_true", default=True)
    parser.add_argument("--no-exit-on-flat", dest="exit_on_flat", action="store_false")

    # Stop
    parser.add_argument("--use-stop", action="store_true", default=False)
    parser.add_argument("--stop-type", type=str, default="ATR", choices=["ATR", "Fixo (%)", "Banda Stop"])
    parser.add_argument("--stop-atr-mult", type=float, default=2.0)
    parser.add_argument("--stop-fixo-pct", type=float, default=2.0)
    parser.add_argument("--stop-band-pct", type=float, default=1.5)

    # Entrada
    parser.add_argument("--use-pullback", action="store_true", default=True)
    parser.add_argument("--no-pullback", dest="use_pullback", action="store_false")
    parser.add_argument("--use-entry-zone", action="store_true", default=False)

    # Capital
    parser.add_argument("--capital", type=float, default=1000.0)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)

    args = parser.parse_args()

    cfg = Config(
        ma_type=args.ma_type,
        ma_length=args.ma_length,
        lookback=args.lookback,
        th_up=args.th_up,
        th_dn=args.th_dn,
        hysteresis=args.hysteresis,
        exit_mode=args.exit_mode,
        pct_up=args.pct_up,
        pct_dn=args.pct_dn,
        alvo_fixo=args.alvo_fixo,
        exit_on_flat=args.exit_on_flat,
        use_stop=args.use_stop,
        stop_type=args.stop_type,
        stop_atr_mult=args.stop_atr_mult,
        stop_fixo_pct=args.stop_fixo_pct,
        stop_band_pct=args.stop_band_pct,
        use_pullback=args.use_pullback,
        use_entry_zone=args.use_entry_zone,
        use_norm=True,
        atr_length=14,
        initial_capital=args.capital,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Carrega dados
    if args.csv:
        df = load_csv(args.csv)
    else:
        df = download_data(args.symbol, args.interval)

    # Executa backtest
    result = run_backtest(df, cfg)

    # Relatório
    print_report(result, cfg)

    # Salva trades em CSV
    if result.trades:
        trades_df = pd.DataFrame([
            {
                "Data Entrada": t.entry_date,
                "Data Saída": t.exit_date,
                "Direção": "LONG" if t.direction == 1 else "SHORT",
                "Tipo": t.comment,
                "Preço Entrada": round(t.entry_price, 4),
                "Preço Saída": round(t.exit_price, 4),
                "P&L (%)": round(t.pnl_pct, 4),
                "Motivo Saída": t.exit_comment,
            }
            for t in result.trades
        ])
        output_path = Path(__file__).parents[1] / "data" / "trades_resultado.csv"
        trades_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n  Trades exportados para: {output_path}")


if __name__ == "__main__":
    main()
