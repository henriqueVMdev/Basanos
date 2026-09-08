"""
Persistência do módulo de automação — SQLite (WAL) em automation/state.db.

Por que SQLite e não o JSON do journal: o runner grava ordens/fills/snapshots
a cada candle concorrendo com as threads HTTP do Flask; WAL dá atomicidade
pós-crash sem dependências novas. Conexão por chamada (thread-safe).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.db")

_DDL = """
CREATE TABLE IF NOT EXISTS deployments (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  strategy_file TEXT NOT NULL,
  params_json TEXT NOT NULL,
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL DEFAULT '15m',
  exchange TEXT NOT NULL DEFAULT 'bybit',
  mode TEXT NOT NULL CHECK(mode IN ('paper','demo','real')),
  account TEXT,                        -- perfil de chaves p/ mode=real: 'prop'|'personal'
  guardrails_json TEXT,                -- {daily_loss_pct, max_loss_pct, check_balance, max_notional}
  status TEXT NOT NULL CHECK(status IN ('created','running','stopped','error')),
  initial_capital REAL NOT NULL,
  equity REAL NOT NULL,
  backtest_ref_json TEXT,
  last_candle_ts INTEGER,
  last_tick_at INTEGER,
  error TEXT,
  created_at INTEGER,
  started_at INTEGER,
  stopped_at INTEGER
);
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  deployment_id TEXT NOT NULL,
  kind TEXT NOT NULL,                 -- 'entry' | 'close'
  side INTEGER NOT NULL,              -- 1 long | -1 short
  type TEXT NOT NULL,                 -- 'limit' | 'market'
  price REAL,
  qty REAL,
  tp_pct REAL, sl_pct REAL, max_bars INTEGER, exposure REAL,
  valid_candle_ts INTEGER,            -- candle (open ts ms) em que a limite vale
  status TEXT NOT NULL,               -- 'working'|'filled'|'expired'|'cancelled'|'rejected'
  exchange_order_id TEXT,
  raw_json TEXT,
  created_at INTEGER, updated_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_orders_dep ON orders(deployment_id, status);
CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  deployment_id TEXT NOT NULL,
  side INTEGER NOT NULL,
  qty REAL NOT NULL,
  exposure REAL NOT NULL,
  entry_price REAL NOT NULL,
  entry_candle_ts INTEGER NOT NULL,
  tp_price REAL, sl_price REAL, max_bars INTEGER,
  entry_fee_rate REAL,
  exit_on_flip INTEGER NOT NULL DEFAULT 0,
  bars_held INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,               -- 'open' | 'closed'
  exit_price REAL, exit_candle_ts INTEGER, exit_reason TEXT,
  pnl_pct REAL, pnl_quote REAL, fees_quote REAL
);
CREATE INDEX IF NOT EXISTS idx_positions_dep ON positions(deployment_id, status);
CREATE TABLE IF NOT EXISTS equity_snapshots (
  deployment_id TEXT NOT NULL,
  candle_ts INTEGER NOT NULL,
  equity REAL NOT NULL,
  PRIMARY KEY (deployment_id, candle_ts)
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  deployment_id TEXT,
  ts INTEGER NOT NULL,
  level TEXT NOT NULL DEFAULT 'info', -- 'info'|'warn'|'error'
  type TEXT NOT NULL,
  message TEXT,
  data_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_dep ON events(deployment_id, id);
"""


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    with _conn() as con:
        con.executescript(_DDL)
        _migrate_real_mode(con)
        for col in ("entry_fee_rate REAL",
                    "exit_on_flip INTEGER NOT NULL DEFAULT 0"):
            try:
                con.execute(f"ALTER TABLE positions ADD COLUMN {col}")
            except sqlite3.OperationalError:      # coluna já existe
                pass


def _migrate_real_mode(con) -> None:
    """Rebuild da tabela deployments p/ incluir mode='real' no CHECK e as
    colunas account/guardrails_json. SQLite não altera CHECK — renomeia,
    recria pelo _DDL e copia. Idempotente: só roda se o schema for antigo."""
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='deployments'"
    ).fetchone()
    if row is None or "'real'" in row["sql"]:
        return
    con.executescript("ALTER TABLE deployments RENAME TO deployments_old;")
    con.executescript(_DDL)          # recria deployments com o schema novo
    con.execute(
        """INSERT INTO deployments
             (id, name, strategy_file, params_json, symbol, interval, exchange,
              mode, status, initial_capital, equity, backtest_ref_json,
              last_candle_ts, last_tick_at, error, created_at, started_at, stopped_at)
           SELECT id, name, strategy_file, params_json, symbol, interval, exchange,
              mode, status, initial_capital, equity, backtest_ref_json,
              last_candle_ts, last_tick_at, error, created_at, started_at, stopped_at
           FROM deployments_old""")
    con.executescript("DROP TABLE deployments_old;")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _row_to_dict(row) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for k in ("params_json", "backtest_ref_json", "raw_json", "data_json",
              "guardrails_json"):
        if k in d and d[k]:
            try:
                d[k[:-5]] = json.loads(d[k])
            except (TypeError, ValueError):
                d[k[:-5]] = None
            del d[k]
    return d


# Deployments
def create_deployment(name, strategy_file, params, symbol, interval, exchange,
                      mode, initial_capital, backtest_ref=None,
                      account=None, guardrails=None) -> str:
    dep_id = uuid.uuid4().hex[:12]
    with _conn() as con:
        con.execute(
            """INSERT INTO deployments
               (id, name, strategy_file, params_json, symbol, interval, exchange,
                mode, account, guardrails_json, status, initial_capital, equity,
                backtest_ref_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,'created',?,?,?,?)""",
            (dep_id, name, strategy_file, json.dumps(params), symbol, interval,
             exchange, mode, account,
             json.dumps(guardrails) if guardrails else None,
             float(initial_capital), float(initial_capital),
             json.dumps(backtest_ref) if backtest_ref else None, _now_ms()))
    return dep_id


def get_deployment(dep_id) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM deployments WHERE id=?", (dep_id,)).fetchone()
    return _row_to_dict(row)


def list_deployments(status=None) -> list:
    q = "SELECT * FROM deployments"
    args = ()
    if status:
        q += " WHERE status=?"
        args = (status,)
    with _conn() as con:
        rows = con.execute(q + " ORDER BY created_at DESC", args).fetchall()
    return [_row_to_dict(r) for r in rows]


def update_deployment(dep_id, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with _conn() as con:
        con.execute(f"UPDATE deployments SET {cols} WHERE id=?",
                    (*fields.values(), dep_id))


def delete_deployment(dep_id) -> None:
    with _conn() as con:
        for table in ("orders", "positions", "equity_snapshots", "events"):
            con.execute(f"DELETE FROM {table} WHERE deployment_id=?", (dep_id,))
        con.execute("DELETE FROM deployments WHERE id=?", (dep_id,))


# Orders
def create_order(dep_id, kind, side, type_, price, qty, valid_candle_ts,
                 tp_pct=None, sl_pct=None, max_bars=None, exposure=None,
                 exchange_order_id=None, raw=None) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO orders (deployment_id, kind, side, type, price, qty,
                 tp_pct, sl_pct, max_bars, exposure, valid_candle_ts, status,
                 exchange_order_id, raw_json, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,'working',?,?,?,?)""",
            (dep_id, kind, side, type_, price, qty, tp_pct, sl_pct, max_bars,
             exposure, valid_candle_ts, exchange_order_id,
             json.dumps(raw) if raw else None, _now_ms(), _now_ms()))
        return cur.lastrowid


def get_working_order(dep_id, kind="entry") -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM orders WHERE deployment_id=? AND kind=? AND status='working' "
            "ORDER BY id DESC LIMIT 1", (dep_id, kind)).fetchone()
    return _row_to_dict(row)


def update_order(order_id, **fields) -> None:
    fields["updated_at"] = _now_ms()
    cols = ", ".join(f"{k}=?" for k in fields)
    with _conn() as con:
        con.execute(f"UPDATE orders SET {cols} WHERE id=?", (*fields.values(), order_id))


# Positions
def open_position(dep_id, side, qty, exposure, entry_price, entry_candle_ts,
                  tp_price, sl_price, max_bars, entry_fee_rate=None,
                  exit_on_flip=0) -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO positions (deployment_id, side, qty, exposure,
                 entry_price, entry_candle_ts, tp_price, sl_price, max_bars,
                 entry_fee_rate, exit_on_flip, bars_held, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,0,'open')""",
            (dep_id, side, qty, exposure, entry_price, entry_candle_ts,
             tp_price, sl_price, max_bars, entry_fee_rate, int(exit_on_flip)))
        return cur.lastrowid


def get_open_position(dep_id) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM positions WHERE deployment_id=? AND status='open' "
            "ORDER BY id DESC LIMIT 1", (dep_id,)).fetchone()
    return _row_to_dict(row)


def update_position(pos_id, **fields) -> None:
    cols = ", ".join(f"{k}=?" for k in fields)
    with _conn() as con:
        con.execute(f"UPDATE positions SET {cols} WHERE id=?", (*fields.values(), pos_id))


def list_closed_positions(dep_id, limit=200) -> list:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM positions WHERE deployment_id=? AND status='closed' "
            "ORDER BY exit_candle_ts DESC LIMIT ?", (dep_id, limit)).fetchall()
    return [_row_to_dict(r) for r in rows]


# Equity / eventos
def add_equity_snapshot(dep_id, candle_ts, equity) -> None:
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO equity_snapshots (deployment_id, candle_ts, equity) "
            "VALUES (?,?,?)", (dep_id, candle_ts, float(equity)))


def get_day_start_equity(dep_id, day_start_ms) -> float | None:
    """Equity de referência p/ o limite de perda diária: primeiro snapshot
    do dia UTC; se o dia ainda não tem snapshot, o último anterior."""
    with _conn() as con:
        row = con.execute(
            "SELECT equity FROM equity_snapshots WHERE deployment_id=? AND "
            "candle_ts>=? ORDER BY candle_ts ASC LIMIT 1",
            (dep_id, day_start_ms)).fetchone()
        if row is None:
            row = con.execute(
                "SELECT equity FROM equity_snapshots WHERE deployment_id=? AND "
                "candle_ts<? ORDER BY candle_ts DESC LIMIT 1",
                (dep_id, day_start_ms)).fetchone()
    return float(row["equity"]) if row else None


def get_equity_curve(dep_id, limit=5000) -> list:
    with _conn() as con:
        rows = con.execute(
            "SELECT candle_ts, equity FROM equity_snapshots WHERE deployment_id=? "
            "ORDER BY candle_ts DESC LIMIT ?", (dep_id, limit)).fetchall()
    return [(r["candle_ts"], r["equity"]) for r in reversed(rows)]


def add_event(dep_id, type_, message, level="info", data=None) -> None:
    with _conn() as con:
        con.execute(
            "INSERT INTO events (deployment_id, ts, level, type, message, data_json) "
            "VALUES (?,?,?,?,?,?)",
            (dep_id, _now_ms(), level, type_, message,
             json.dumps(data) if data else None))


def list_events(dep_id, limit=100) -> list:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM events WHERE deployment_id=? ORDER BY id DESC LIMIT ?",
            (dep_id, limit)).fetchall()
    return [_row_to_dict(r) for r in rows]
