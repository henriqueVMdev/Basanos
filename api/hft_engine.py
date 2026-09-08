"""
HFT on-chain (memecoins) — motor de execução rápida em loop, sem LLM no
caminho crítico. Os agentes de IA orquestram (configuram/ligam/desligam);
o motor decide tick a tick por regras.

Latência real: ~2-5s por tick (piso das APIs públicas GeckoTerminal/
DexScreener). Sub-segundo exigiria RPC dedicado/websocket — fora do escopo.

Modos:
  paper  — fills simulados com slippage proporcional ao impacto na liquidez
           + taxa de DEX. Default e único modo implementado.
  real   — NÃO implementado. Exigiria WALLET_PRIVATE_KEY + RPC da chain e
           assinatura de swaps. Deixado explícito para decisão futura.

Estado em data/hft_state.json.
"""

import json
import threading
import time
from pathlib import Path

import requests
from flask import Blueprint, jsonify, request

hft_bp = Blueprint("hft", __name__)

STATE_FILE = Path(__file__).parents[1] / "data" / "hft_state.json"
_lock = threading.RLock()
_thread = None
_stop_flag = threading.Event()

GT = "https://api.geckoterminal.com/api/v2"
DS = "https://api.dexscreener.com"

DEFAULT_CONFIG = {
    "chain": "robinhood",          # slug GeckoTerminal
    "mode": "paper",
    "scan_interval_s": 60,          # busca de candidatos (trending)
    "tick_interval_s": 3,           # loop rápido sobre a watchlist
    # Entrada
    "min_liquidity_usd": 50_000,    # skill do usuário: < $50k = rugpull
    "min_pool_age_h": 24,           # pools mais novos = risco alto
    "min_buy_ratio": 0.62,          # buys/(buys+sells) na última hora
    "min_vol_accel": 1.5,           # vol_m5*288 vs vol_h24 (ritmo 5min anualizado no dia)
    # Saída
    "take_profit_pct": 12.0,
    "stop_loss_pct": 6.0,
    "max_hold_min": 45,
    # Sizing e guardrails
    "position_usd": 100.0,
    "max_positions": 3,
    "daily_loss_stop_usd": 150.0,   # para o motor no dia se perder isso
    "fee_pct": 0.3,                 # taxa DEX por perna
}


def _now():
    return time.time()


def _load():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "config": dict(DEFAULT_CONFIG),
        "running": False,
        "capital": 1000.0,
        "initial_capital": 1000.0,
        "positions": {},   # token_addr -> position
        "watchlist": {},   # token_addr -> {symbol, pool, liq, added_at}
        "trades": [],
        "events": [],
        "day": "",
        "day_pnl": 0.0,
        "halted": False,
    }


def _save(st):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(st, indent=1, ensure_ascii=False))


def _event(st, kind, text):
    st["events"].append({"ts": _now(), "kind": kind, "text": text})
    st["events"] = st["events"][-200:]


def _slippage_pct(order_usd, liquidity_usd):
    """Impacto estimado: proporcional ao tamanho vs liquidez do pool."""
    if not liquidity_usd:
        return 5.0
    return min(5.0, 100.0 * order_usd / liquidity_usd * 2)


# ─── Scan: candidatos via GeckoTerminal trending ────────────────────────────

def _scan(st):
    cfg = st["config"]
    try:
        r = requests.get(f"{GT}/networks/{cfg['chain']}/trending_pools",
                         params={"include": "base_token"}, timeout=15)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        _event(st, "warn", f"Scan falhou: {e}")
        return

    included = {i["id"]: i["attributes"] for i in payload.get("included", [])
                if i.get("type") == "token"}
    added = 0
    for pool in payload.get("data", []):
        a = pool.get("attributes", {})
        rel = pool.get("relationships", {})
        base = included.get(rel.get("base_token", {}).get("data", {}).get("id", ""), {})
        addr = base.get("address")
        if not addr or addr in st["watchlist"] or addr in st["positions"]:
            continue
        try:
            liq = float(a.get("reserve_in_usd") or 0)
            created = a.get("pool_created_at")
            age_h = ((_now() - time.mktime(time.strptime(
                created[:19], "%Y-%m-%dT%H:%M:%S"))) / 3600) if created else 0
        except Exception:
            continue
        if liq < cfg["min_liquidity_usd"] or age_h < cfg["min_pool_age_h"]:
            continue
        st["watchlist"][addr] = {"symbol": base.get("symbol", "?"),
                                 "pool": a.get("address"),
                                 "liq": liq, "added_at": _now()}
        added += 1
    # Expira watchlist velha (>2h sem posição)
    cutoff = _now() - 7200
    st["watchlist"] = {k: v for k, v in st["watchlist"].items()
                       if v["added_at"] > cutoff or k in st["positions"]}
    if added:
        _event(st, "scan", f"{added} candidatos novos na watchlist "
                           f"({len(st['watchlist'])} total)")


# ─── Tick: dados frescos via DexScreener batch ──────────────────────────────

DS_SLUGS = {"eth": "ethereum"}


def _fetch_batch(chain, addrs):
    slug = DS_SLUGS.get(chain, chain)
    out = {}
    for i in range(0, len(addrs), 30):
        chunk = ",".join(addrs[i:i + 30])
        try:
            r = requests.get(f"{DS}/tokens/v1/{slug}/{chunk}", timeout=10)
            r.raise_for_status()
            for pair in r.json():
                base = pair.get("baseToken", {}).get("address", "")
                cur = out.get(base)
                liq = (pair.get("liquidity") or {}).get("usd") or 0
                if not cur or liq > cur["_liq"]:
                    txns_h1 = (pair.get("txns") or {}).get("h1", {}) or {}
                    vol = pair.get("volume") or {}
                    out[base] = {
                        "_liq": liq,
                        "price": float(pair.get("priceUsd") or 0),
                        "liq": liq,
                        "vol_m5": vol.get("m5") or 0,
                        "vol_h24": vol.get("h24") or 0,
                        "buys_h1": txns_h1.get("buys") or 0,
                        "sells_h1": txns_h1.get("sells") or 0,
                    }
        except Exception:
            continue
    return out


def _try_enter(st, addr, info, snap):
    cfg = st["config"]
    if st["halted"] or len(st["positions"]) >= cfg["max_positions"]:
        return
    if snap["price"] <= 0 or snap["liq"] < cfg["min_liquidity_usd"]:
        return
    total_h1 = snap["buys_h1"] + snap["sells_h1"]
    if total_h1 < 20:
        return
    buy_ratio = snap["buys_h1"] / total_h1
    accel = (snap["vol_m5"] * 288 / snap["vol_h24"]) if snap["vol_h24"] else 0
    if buy_ratio < cfg["min_buy_ratio"] or accel < cfg["min_vol_accel"]:
        return

    usd = min(cfg["position_usd"], st["capital"])
    if usd < 10:
        return
    slip = _slippage_pct(usd, snap["liq"])
    fill = snap["price"] * (1 + slip / 100)
    fee = usd * cfg["fee_pct"] / 100
    qty = (usd - fee) / fill
    st["capital"] -= usd
    st["positions"][addr] = {
        "symbol": info["symbol"], "qty": qty, "entry": fill,
        "usd_in": usd, "opened_at": _now(),
        "peak": fill, "slip_in": slip,
    }
    _event(st, "entry", f"COMPRA {info['symbol']} ${usd:.0f} @ {fill:.8g} "
                        f"(slip {slip:.2f}%, buy_ratio {buy_ratio:.0%}, "
                        f"accel {accel:.1f}x)")


def _try_exit(st, addr, pos, snap, force_reason=None):
    cfg = st["config"]
    price = snap["price"] if snap else pos["entry"]
    if price <= 0:
        return
    pos["peak"] = max(pos["peak"], price)
    pnl_pct = (price / pos["entry"] - 1) * 100
    held_min = (_now() - pos["opened_at"]) / 60

    reason = force_reason
    if not reason:
        if pnl_pct >= cfg["take_profit_pct"]:
            reason = "take_profit"
        elif pnl_pct <= -cfg["stop_loss_pct"]:
            reason = "stop_loss"
        elif held_min >= cfg["max_hold_min"]:
            reason = "timeout"
    if not reason:
        return

    liq = snap["liq"] if snap else 0
    gross = pos["qty"] * price
    slip = _slippage_pct(gross, liq)
    net = gross * (1 - slip / 100) * (1 - cfg["fee_pct"] / 100)
    pnl = net - pos["usd_in"]
    st["capital"] += net
    st["day_pnl"] += pnl
    st["trades"].append({
        "symbol": pos["symbol"], "token": addr,
        "entry": pos["entry"], "exit": price,
        "usd_in": pos["usd_in"], "usd_out": round(net, 2),
        "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
        "held_min": round(held_min, 1), "reason": reason,
        "closed_at": _now(),
    })
    st["trades"] = st["trades"][-500:]
    del st["positions"][addr]
    _event(st, "exit", f"VENDA {pos['symbol']} @ {price:.8g} | "
                       f"PnL ${pnl:+.2f} ({pnl_pct:+.1f}%) [{reason}]")

    if st["day_pnl"] <= -cfg["daily_loss_stop_usd"]:
        st["halted"] = True
        _event(st, "guardrail", f"STOP DIÁRIO: perda de ${-st['day_pnl']:.2f} "
                                "— sem novas entradas até amanhã")


def _tick(st):
    cfg = st["config"]
    today = time.strftime("%Y-%m-%d")
    if st["day"] != today:
        st["day"] = today
        st["day_pnl"] = 0.0
        st["halted"] = False

    addrs = list(set(list(st["watchlist"]) + list(st["positions"])))
    if not addrs:
        return
    snaps = _fetch_batch(cfg["chain"], addrs)

    for addr, pos in list(st["positions"].items()):
        _try_exit(st, addr, pos, snaps.get(addr))
    for addr, info in list(st["watchlist"].items()):
        if addr not in st["positions"] and addr in snaps:
            _try_enter(st, addr, info, snaps[addr])


# ─── Loop principal (thread) ─────────────────────────────────────────────────

def _run_loop():
    last_scan = 0.0
    while not _stop_flag.is_set():
        try:
            with _lock:
                st = _load()
                if not st["running"]:
                    break
                if _now() - last_scan >= st["config"]["scan_interval_s"]:
                    _scan(st)
                    last_scan = _now()
                _tick(st)
                _save(st)
        except Exception as e:
            with _lock:
                st = _load()
                _event(st, "error", f"Loop: {e}")
                _save(st)
        _stop_flag.wait(st["config"]["tick_interval_s"])


def _ensure_thread():
    global _thread
    if _thread is None or not _thread.is_alive():
        _stop_flag.clear()
        _thread = threading.Thread(target=_run_loop, daemon=True,
                                   name="hft-engine")
        _thread.start()


# ─── API ─────────────────────────────────────────────────────────────────────

@hft_bp.get("/api/hft/status")
def hft_status():
    with _lock:
        st = _load()
    open_value = sum(p["usd_in"] for p in st["positions"].values())
    return jsonify({
        "running": st["running"], "halted": st["halted"],
        "mode": st["config"]["mode"], "chain": st["config"]["chain"],
        "capital": round(st["capital"], 2),
        "initial_capital": st["initial_capital"],
        "equity_est": round(st["capital"] + open_value, 2),
        "day_pnl": round(st["day_pnl"], 2),
        "positions": st["positions"], "watchlist_size": len(st["watchlist"]),
        "trades": st["trades"][-50:], "events": st["events"][-50:],
        "config": st["config"],
        "n_trades": len(st["trades"]),
        "thread_alive": _thread.is_alive() if _thread else False,
    })


@hft_bp.post("/api/hft/start")
def hft_start():
    with _lock:
        st = _load()
        if st["config"]["mode"] != "paper":
            return jsonify({"error": "Somente modo paper implementado. Execução "
                                     "real exige carteira + RPC (decisão manual)."}), 400
        st["running"] = True
        _event(st, "state", "Motor HFT ligado (paper)")
        _save(st)
    _ensure_thread()
    return jsonify({"ok": True})


@hft_bp.post("/api/hft/stop")
def hft_stop():
    close = bool((request.get_json(silent=True) or {}).get("close_positions"))
    with _lock:
        st = _load()
        st["running"] = False
        if close:
            snaps = _fetch_batch(st["config"]["chain"], list(st["positions"]))
            for addr, pos in list(st["positions"].items()):
                _try_exit(st, addr, pos, snaps.get(addr), force_reason="manual_stop")
        _event(st, "state", "Motor HFT desligado"
               + (" (posições fechadas)" if close else ""))
        _save(st)
    _stop_flag.set()
    return jsonify({"ok": True})


@hft_bp.post("/api/hft/config")
def hft_config():
    body = request.get_json(force=True) or {}
    if body.get("mode") == "real":
        return jsonify({"error": "Modo real não implementado — exige "
                                 "WALLET_PRIVATE_KEY/RPC e decisão manual."}), 400
    with _lock:
        st = _load()
        for k, v in body.items():
            if k in DEFAULT_CONFIG and k != "mode":
                st["config"][k] = type(DEFAULT_CONFIG[k])(v)
        _event(st, "config", f"Config alterada: {body}")
        _save(st)
        return jsonify(st["config"])


@hft_bp.post("/api/hft/reset")
def hft_reset():
    with _lock:
        st = _load()
        if st["running"]:
            return jsonify({"error": "Pare o motor antes de resetar"}), 400
        capital = float((request.get_json(silent=True) or {}).get("capital", 1000))
        st = {
            "config": st["config"], "running": False,
            "capital": capital, "initial_capital": capital,
            "positions": {}, "watchlist": {}, "trades": [], "events": [],
            "day": "", "day_pnl": 0.0, "halted": False,
        }
        _event(st, "state", f"Reset: capital ${capital:.0f}")
        _save(st)
    return jsonify({"ok": True})
