"""
Analytics de derivativos: Black-Scholes (preço teórico + greeks), chain com
greeks, superfície de volatilidade (vários vencimentos) e simulador de
estratégias com opções (payoff, breakevens, greeks líquidos e matriz de
cenários p/ monitoramento de risco).

Convenções: theta por dia corrido; vega por 1 ponto de vol (1%); rho por
1 pp de juros. r vem da treasury 3M (markets_data.yield_curve).
"""

from __future__ import annotations

import math
import time

_CACHE: dict = {}


def _cached(key, ttl_s, fn):
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl_s:
        return hit[1]
    data = fn()
    _CACHE[key] = (time.time(), data)
    return data


def _f(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _n_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _n_pdf(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def bs(kind: str, s, k, t, sigma, r=0.04, q=0.0) -> dict:
    """Preço e greeks Black-Scholes. t em anos; sigma decimal (0.25 = 25%)."""
    if not s or not k or t is None or not sigma or s <= 0 or k <= 0 or sigma <= 0:
        return {}
    t = max(t, 1e-6)
    sq = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + sigma * sigma / 2) * t) / sq
    d2 = d1 - sq
    disc_r, disc_q = math.exp(-r * t), math.exp(-q * t)
    if kind == "call":
        price = s * disc_q * _n_cdf(d1) - k * disc_r * _n_cdf(d2)
        delta = disc_q * _n_cdf(d1)
        theta = (-s * disc_q * _n_pdf(d1) * sigma / (2 * math.sqrt(t))
                 - r * k * disc_r * _n_cdf(d2) + q * s * disc_q * _n_cdf(d1))
        rho = k * t * disc_r * _n_cdf(d2) / 100
    else:
        price = k * disc_r * _n_cdf(-d2) - s * disc_q * _n_cdf(-d1)
        delta = -disc_q * _n_cdf(-d1)
        theta = (-s * disc_q * _n_pdf(d1) * sigma / (2 * math.sqrt(t))
                 + r * k * disc_r * _n_cdf(-d2) - q * s * disc_q * _n_cdf(-d1))
        rho = -k * t * disc_r * _n_cdf(-d2) / 100
    gamma = disc_q * _n_pdf(d1) / (s * sq)
    vega = s * disc_q * _n_pdf(d1) * math.sqrt(t) / 100
    return {"price": round(price, 4), "delta": round(delta, 4),
            "gamma": round(gamma, 6), "theta": round(theta / 365, 4),
            "vega": round(vega, 4), "rho": round(rho, 4)}


def _risk_free() -> float:
    try:
        from providers import markets_data
        pts = {p["label"]: p["now"] for p in markets_data.yield_curve()["points"]}
        v = pts.get("3M")
        return v / 100 if v else 0.04
    except Exception:
        return 0.04


def add_greeks(chain: dict) -> dict:
    """Anexa greeks BS a cada linha da chain de markets_data.option_chain."""
    s = chain.get("spot")
    dte = chain.get("dte")
    if not s or dte is None or chain.get("error"):
        return chain
    t = max(dte, 0.5) / 365
    r = _risk_free()
    for kind, key in (("call", "calls"), ("put", "puts")):
        for row in chain.get(key, []):
            g = bs(kind, s, row.get("strike"), t, row.get("iv"), r)
            row.update({k: g.get(k) for k in
                        ("delta", "gamma", "theta", "vega", "rho", "price")})
            row["theo"] = row.pop("price", None)
    chain["risk_free"] = round(r * 100, 2)
    return chain


# ── superfície de volatilidade ───────────────────────────────────────────

_MONEYNESS = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]


def vol_surface(symbol: str, max_expiries: int = 6) -> dict:
    from providers import markets_data

    def fetch():
        first = markets_data.option_chain(symbol)
        if first.get("error"):
            return first
        spot = first["spot"]
        expiries = first["expiries"][:max_expiries]
        grid, dtes = [], []
        for exp in expiries:
            ch = (first if exp == first["expiry"]
                  else markets_data.option_chain(symbol, exp))
            if ch.get("error"):
                continue
            dtes.append(ch["dte"])
            row = []
            for m in _MONEYNESS:
                k_target = spot * m
                # lado OTM: puts abaixo do spot, calls acima (padrão de mercado)
                side = ch["puts"] if m < 1.0 else ch["calls"]
                cands = [x for x in side if x.get("iv") and x.get("strike")]
                if not cands:
                    row.append(None)
                    continue
                best = min(cands, key=lambda x: abs(x["strike"] - k_target))
                # descarta se o strike mais próximo está longe demais (>4% do alvo)
                row.append(round(best["iv"] * 100, 2)
                           if abs(best["strike"] - k_target) / k_target < 0.04 else None)
            grid.append(row)
        return {"symbol": symbol.upper(), "spot": spot,
                "moneyness": _MONEYNESS, "expiries": expiries[:len(grid)],
                "dtes": dtes, "iv_grid": grid, "ts": int(time.time() * 1000)}

    return _cached(("surface", symbol.upper(), max_expiries), 900, fetch)


# ── simulador de estratégias ─────────────────────────────────────────────

def strategy_eval(payload: dict) -> dict:
    """
    payload: {spot, dte, iv_default?, legs: [{kind: call|put|stock,
              side: buy|sell, strike?, qty, premium, iv?}]}
    Retorna payoff no vencimento, P&L teórico hoje, breakevens, greeks
    líquidos e matriz de cenários (spot x tempo).
    """
    spot = _f(payload.get("spot"))
    dte = _f(payload.get("dte")) or 30
    legs = payload.get("legs") or []
    if not spot or not legs:
        raise ValueError("spot e legs obrigatórios")
    r = _risk_free()
    iv_default = (_f(payload.get("iv_default")) or 30) / 100
    t = max(dte, 0.5) / 365

    def leg_val(leg, s, tt):
        kind = leg.get("kind")
        qty = _f(leg.get("qty")) or 1
        sign = 1 if leg.get("side") == "buy" else -1
        if kind == "stock":
            return sign * qty * s
        k = _f(leg.get("strike"))
        if tt <= 0:  # payoff no vencimento
            intrinsic = max(s - k, 0) if kind == "call" else max(k - s, 0)
            return sign * qty * intrinsic
        iv = (_f(leg.get("iv")) or 0) / 100 or iv_default
        g = bs(kind, s, k, tt, iv, r)
        return sign * qty * (g.get("price") or 0)

    def leg_cost(leg):
        qty = _f(leg.get("qty")) or 1
        sign = 1 if leg.get("side") == "buy" else -1
        base = _f(leg.get("premium"))
        if leg.get("kind") == "stock":
            base = spot if base is None else base
        return sign * qty * (base or 0)

    cost = sum(leg_cost(l) for l in legs)

    # grade de preços ±30%
    n = 121
    s_grid = [spot * (0.70 + 0.60 * i / (n - 1)) for i in range(n)]
    payoff = [round(sum(leg_val(l, s, 0) for l in legs) - cost, 4) for s in s_grid]
    pnl_now = [round(sum(leg_val(l, s, t) for l in legs) - cost, 4) for s in s_grid]

    # breakevens por mudança de sinal do payoff
    breakevens = []
    for i in range(1, n):
        a, b = payoff[i - 1], payoff[i]
        if a == 0 or (a < 0 < b) or (b < 0 < a):
            x = s_grid[i - 1] + (s_grid[i] - s_grid[i - 1]) * (0 - a) / (b - a) \
                if b != a else s_grid[i]
            breakevens.append(round(x, 2))

    # greeks líquidos hoje
    net = {k: 0.0 for k in ("delta", "gamma", "theta", "vega", "rho")}
    theo_total = 0.0
    legs_out = []
    for leg in legs:
        sign = 1 if leg.get("side") == "buy" else -1
        qty = _f(leg.get("qty")) or 1
        if leg.get("kind") == "stock":
            net["delta"] += sign * qty
            legs_out.append({**leg, "theo": round(spot, 4)})
            continue
        iv = (_f(leg.get("iv")) or 0) / 100 or iv_default
        g = bs(leg["kind"], spot, _f(leg.get("strike")), t, iv, r)
        for k in net:
            net[k] += sign * qty * (g.get(k) or 0)
        theo_total += sign * qty * (g.get("price") or 0)
        legs_out.append({**leg, "theo": g.get("price"),
                         "delta": g.get("delta"), "iv_used": round(iv * 100, 1)})
    net = {k: round(v, 4) for k, v in net.items()}

    # cenários: spot x tempo restante
    spot_shifts = [-0.10, -0.05, 0, 0.05, 0.10]
    time_steps = [("hoje", t), ("metade", t / 2), ("vencimento", 0)]
    scenarios = []
    for lbl, tt in time_steps:
        row = []
        for sh in spot_shifts:
            s = spot * (1 + sh)
            row.append(round(sum(leg_val(l, s, tt) for l in legs) - cost, 2))
        scenarios.append({"when": lbl, "pnl": row})

    mx, mn = max(payoff), min(payoff)
    return {
        "spot": spot, "dte": dte, "risk_free": round(r * 100, 2),
        "cost": round(cost, 4),
        "s_grid": [round(s, 4) for s in s_grid],
        "payoff": payoff, "pnl_now": pnl_now,
        "breakevens": breakevens,
        "max_gain": round(mx, 2), "max_loss": round(mn, 2),
        "unbounded_gain": payoff[-1] > payoff[-2] + 1e-9,
        "unbounded_loss": payoff[0] < payoff[1] - 1e-9 or payoff[-1] < payoff[-2] - 1e-9,
        "net_greeks": net, "theo_total": round(theo_total, 4),
        "legs": legs_out,
        "spot_shifts": spot_shifts,
        "scenarios": scenarios,
    }
