"""Dados de mercado (OHLCV) via CCXT — read-only, dados públicos.

Fonte unificada de candles cripto para as exchanges suportadas:
Binance (perp USDⓈ-M), Bybit, OKX e Hyperliquid.

Princípios:
- Somente endpoints PÚBLICOS. Nenhuma apiKey/secret é passada — sem permissão de
  trade/saque ao alcance deste código (mesma postura de costs/funding.py).
- Saída drop-in com o resto do projeto: DataFrame com colunas
  Open/High/Low/Close/Volume e índice DatetimeIndex (UTC, tz-naive), ordenado.
  Mesmo formato que _download_data_safe (yfinance) devolve.
- Confiar no candle que o endpoint retorna; não recalcular.

Uso:
    from providers.market_data import fetch_ohlcv, SUPPORTED_EXCHANGES
    df = fetch_ohlcv("BTC", "1h", exchange="bybit", total=2000)
"""

from __future__ import annotations

import pandas as pd

# Nome amigável -> classe CCXT + convenção de liquidação (settle) dos perps.
# binance usa o cliente binanceusdm (futuros USDⓈ-M). Hyperliquid liquida em USDC.
_REGISTRY: dict[str, dict] = {
    "binance":     {"ccxt": "binanceusdm", "settle": "USDT"},
    "bybit":       {"ccxt": "bybit",       "settle": "USDT"},
    "okx":         {"ccxt": "okx",         "settle": "USDT"},
    "hyperliquid": {"ccxt": "hyperliquid", "settle": "USDC"},
}

SUPPORTED_EXCHANGES = tuple(_REGISTRY.keys())

# Colunas de saída (capitalizadas, como o restante do projeto espera).
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Timeframes que resamplamos a partir de uma base quando a exchange não os
# oferece nativamente (ex.: alguns perps não expõem 2h/4h direto).
_RESAMPLE_FROM = {"2h": "1h", "4h": "1h"}

# Cache de instâncias CCXT — load_markets() é caro, reusar por processo.
_EXCHANGE_CACHE: dict[str, object] = {}


def get_exchange(name: str):
    """Instância CCXT pública (sem credenciais) para a exchange dada.

    Cacheada por nome. `name` é o nome amigável (ver SUPPORTED_EXCHANGES).
    """
    import ccxt

    name = name.lower().strip()
    if name not in _REGISTRY:
        raise ValueError(
            f"exchange não suportada: {name!r}. "
            f"Suportadas: {', '.join(SUPPORTED_EXCHANGES)}"
        )
    if name in _EXCHANGE_CACHE:
        return _EXCHANGE_CACHE[name]

    cls = getattr(ccxt, _REGISTRY[name]["ccxt"])
    ex = cls({
        "enableRateLimit": True,
        "timeout": 30000,
        # swap = perpétuo; sem efeito em exchanges que ignoram a opção.
        "options": {"defaultType": "swap"},
    })
    _EXCHANGE_CACHE[name] = ex
    return ex


def normalize_symbol(symbol: str, exchange: str) -> str:
    """Converte vários formatos de símbolo no símbolo unificado CCXT do perp.

    Aceita 'BTC', 'BTC-USD', 'BTCUSDT', 'BTC/USDT', 'BTC/USDT:USDT'.
    Devolve, p.ex., 'BTC/USDT:USDT' (binance/bybit/okx) ou 'BTC/USDC:USDC'
    (hyperliquid). Se já vier no formato unificado com settle (contém ':'),
    é repassado intacto.
    """
    exchange = exchange.lower().strip()
    settle = _REGISTRY[exchange]["settle"]
    s = symbol.strip().upper()

    # Já é símbolo unificado com settle (ex.: 'BTC/USDC:USDC') — não mexer.
    if ":" in s:
        return s

    # Extrai a moeda-base de qualquer formato comum.
    if "/" in s:                       # 'BTC/USDT'
        base = s.split("/")[0]
    elif "-" in s:                     # 'BTC-USD' (estilo yfinance)
        base = s.split("-")[0]
    elif s.endswith("USDT"):           # 'BTCUSDT'
        base = s[:-4]
    elif s.endswith("USDC"):           # 'BTCUSDC'
        base = s[:-4]
    elif s.endswith("USD"):            # 'BTCUSD'
        base = s[:-3]
    else:                              # 'BTC'
        base = s

    return f"{base}/{settle}:{settle}"


def fetch_ohlcv(
    symbol: str = "BTC",
    timeframe: str = "1h",
    exchange: str = "binance",
    limit: int = 1000,
    total: int | None = None,
) -> pd.DataFrame:
    """Candles OHLCV de uma exchange como DataFrame pronto pro backtest.

    Args:
        symbol: base ou símbolo em qualquer formato comum (ver normalize_symbol).
        timeframe: '15m', '30m', '1h', '2h', '4h', '1d', ...
        exchange: uma de SUPPORTED_EXCHANGES.
        limit: candles por requisição (e total, quando `total` é None).
        total: se dado, pagina via `since` até reunir ~`total` candles
            (histórico longo p/ backtest). Resultado deduplicado e ordenado.

    Returns:
        DataFrame [Open, High, Low, Close, Volume], índice DatetimeIndex UTC
        tz-naive, ordenado crescente. A última linha pode ser a vela em formação.
    """
    ex = get_exchange(exchange)
    sym = normalize_symbol(symbol, exchange)

    # Resolve timeframe: nativo, ou resample a partir de uma base suportada.
    base_tf, resample_rule = _resolve_timeframe(ex, timeframe)

    if total and total > limit:
        raw = _paginate(ex, sym, base_tf, limit, total)
    else:
        raw = ex.fetch_ohlcv(sym, timeframe=base_tf, limit=limit)

    df = _to_df(raw)
    if resample_rule:
        df = _resample(df, resample_rule)
    return df


def _resolve_timeframe(ex, timeframe: str) -> tuple[str, str | None]:
    """Devolve (timeframe_para_buscar, regra_resample|None).

    Se a exchange suporta `timeframe` nativamente, usa direto. Senão, se há um
    caminho de resample conhecido (2h/4h <- 1h) e a base é suportada, usa a base.
    """
    supported = getattr(ex, "timeframes", None) or {}
    if not supported or timeframe in supported:
        return timeframe, None
    base = _RESAMPLE_FROM.get(timeframe)
    if base and base in supported:
        return base, timeframe
    # Sem caminho conhecido — tenta nativo e deixa a exchange reclamar.
    return timeframe, None


def _paginate(ex, symbol: str, timeframe: str, limit: int, total: int) -> list:
    """Pagina via `since` até reunir ~`total` candles (ou esgotar o histórico)."""
    ms = ex.parse_timeframe(timeframe) * 1000
    since = ex.milliseconds() - total * ms
    lotes: list[list] = []
    obtidas = 0
    while obtidas < total:
        lote = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not lote:
            break
        lotes.append(lote)
        obtidas += len(lote)
        since = lote[-1][0] + ms
        if len(lote) < 2:  # fim do histórico disponível
            break
    return [linha for lote in lotes for linha in lote]


def _to_df(raw: list) -> pd.DataFrame:
    """Lista CCXT [ts, o, h, l, c, v] -> DataFrame padrão do projeto."""
    if not raw:
        raise ValueError("Nenhum candle retornado pela exchange.")
    df = pd.DataFrame(raw, columns=["ts", "Open", "High", "Low", "Close", "Volume"])
    idx = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.drop(columns=["ts"])
    df.index = pd.DatetimeIndex(idx, name="Date")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[_OHLCV_COLS]


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Agrega candles para um timeframe maior (ex.: 1h -> 4h)."""
    out = df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return out
