"""
Estratégia SMA Crossover — Backtest Engine (uso standalone / CLI).

Interface clara para ser substituída por qualquer outra estratégia.
Suporta custos de transação e slippage configuráveis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Classe de configuração
@dataclass
class SMACrossConfig:
    fast:             int   = 20        # período da MA rápida
    slow:             int   = 50        # período da MA lenta
    initial_capital:  float = 10_000.0
    commission_pct:   float = 0.001     # 0.1% por trade
    slippage_pct:     float = 0.0005    # 0.05% slippage


# Resultado
@dataclass
class BacktestResult:
    trades:       list[dict]   = field(default_factory=list)
    equity_curve: dict         = field(default_factory=dict)  # {dates, values}
    metrics:      dict         = field(default_factory=dict)


# Estratégia base (interface)
class Strategy:
    """Interface mínima que qualquer estratégia deve implementar."""

    def run(self, df: pd.DataFrame, cfg) -> BacktestResult:
        raise NotImplementedError


# SMA Crossover
class SMACrossover(Strategy):
    """
    SMA Crossover: compra quando MA rápida cruza acima da lenta; vende ao contrário.
    Apenas operações Long para simplificação.
    """

    def run(self, df: pd.DataFrame, cfg: SMACrossConfig = None) -> BacktestResult:
        if cfg is None:
            cfg = SMACrossConfig()

        df = df.copy()
        df["ma_fast"] = df["Close"].rolling(cfg.fast).mean()
        df["ma_slow"] = df["Close"].rolling(cfg.slow).mean()
        df = df.dropna(subset=["ma_fast", "ma_slow"]).reset_index()

        equity        = cfg.initial_capital
        position      = False
        entry_price   = 0.0
        entry_date    = ""
        trades        = []
        equity_series = []
        dates_series  = []

        for i in range(1, len(df)):
            row      = df.iloc[i]
            prev     = df.iloc[i - 1]
            close    = float(row["Close"])
            date_str = str(row.get("Date", row.get("index", i)))[:10]

            # Sinal de entrada: cruzamento para cima
            cross_up   = (prev["ma_fast"] <= prev["ma_slow"]) and (row["ma_fast"] > row["ma_slow"])
            # Sinal de saída: cruzamento para baixo
            cross_down = (prev["ma_fast"] >= prev["ma_slow"]) and (row["ma_fast"] < row["ma_slow"])

            if cross_up and not position:
                # Aplica slippage na entrada
                exec_price = close * (1.0 + cfg.slippage_pct)
                equity    *= (1.0 - cfg.commission_pct)
                entry_price = exec_price
                entry_date  = date_str
                position    = True

            elif cross_down and position:
                # Aplica slippage na saída
                exec_price = close * (1.0 - cfg.slippage_pct)
                equity    *= (1.0 - cfg.commission_pct)
                pnl_pct    = (exec_price / entry_price - 1.0) * 100.0
                equity    *= (1.0 + pnl_pct / 100.0)
                trades.append({
                    "entry_date":  entry_date,
                    "exit_date":   date_str,
                    "entry_price": round(entry_price, 6),
                    "exit_price":  round(exec_price,  6),
                    "pnl_pct":     round(pnl_pct,     6),
                    "side":        "long",
                })
                position = False

            equity_series.append(round(equity, 4))
            dates_series.append(date_str)

        # Fecha posição aberta no último candle
        if position and len(df) > 0:
            last_close  = float(df.iloc[-1]["Close"]) * (1.0 - cfg.slippage_pct)
            pnl_pct     = (last_close / entry_price - 1.0) * 100.0
            equity     *= (1.0 - cfg.commission_pct)
            equity     *= (1.0 + pnl_pct / 100.0)
            trades.append({
                "entry_date":  entry_date,
                "exit_date":   str(df.iloc[-1].get("Date", ""))[:10],
                "entry_price": round(entry_price, 6),
                "exit_price":  round(last_close,  6),
                "pnl_pct":     round(pnl_pct,     6),
                "side":        "long",
            })
            equity_series[-1] = round(equity, 4)

        return BacktestResult(
            trades=trades,
            equity_curve={"dates": dates_series, "values": equity_series},
            metrics=_compute_metrics(trades, equity_series, cfg.initial_capital),
        )


# Métricas
def _safe(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _compute_metrics(trades: list[dict], equity: list[float], ic: float) -> dict:
    if not trades or not equity:
        return {}

    eq   = np.array(equity, dtype=float)
    rets = np.diff(eq) / eq[:-1]
    sharpe = (float(rets.mean() / rets.std() * np.sqrt(252))
              if len(rets) > 1 and rets.std() > 0 else 0.0)

    # Drawdown
    peak   = np.maximum.accumulate(eq)
    dd_arr = (eq - peak) / peak * 100
    max_dd = float(dd_arr.min())

    pnls   = [t["pnl_pct"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate   = len(wins) / len(pnls) * 100 if pnls else 0.0
    avg_win    = float(np.mean(wins))  if wins   else 0.0
    avg_loss   = float(np.mean(losses)) if losses else 0.0
    pf         = (sum(wins) / abs(sum(losses))
                  if losses and sum(losses) != 0 else None)
    expectancy = win_rate / 100 * avg_win + (1 - win_rate / 100) * avg_loss

    return {
        "final_equity":   _safe(float(eq[-1])),
        "total_return":   _safe(float((eq[-1] / ic - 1) * 100)),
        "sharpe":         _safe(sharpe),
        "max_dd":         _safe(max_dd),
        "win_rate":       _safe(win_rate),
        "profit_factor":  _safe(pf),
        "total_trades":   len(trades),
        "avg_win":        _safe(avg_win),
        "avg_loss":       _safe(avg_loss),
        "expectancy":     _safe(expectancy),
        "initial_capital": ic,
    }


# Download de dados
def download_data(
    ticker: str = "BTC-USD",
    period: str = "4y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Baixa dados OHLCV via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("Instale yfinance: pip install yfinance")

    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"Sem dados para '{ticker}'")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
