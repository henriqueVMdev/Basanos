# AGENTS.md

## Cursor Cloud specific instructions

Standard setup/run/test commands live in `README.md` and `docs/OPERATIONS.md`. The notes below capture only the non-obvious, environment-specific details for this repo. The startup update script already creates the Python venv, installs backend deps (`requirements.txt` + `pytest`), and runs `npm ci` in `frontend/`.

### Services

- **Backend (Flask API, required)** — run `.venv/bin/python server.py`. Serves on `http://localhost:5000`, all routes under `/api/...`. Runs with `debug=True` and the watchdog reloader, so it hot-reloads on `.py` edits.
- **Frontend (Vue 3 + Vite, required for UI)** — run `npm run dev` in `frontend/`. Serves on `http://localhost:5173` and proxies `/api` → `http://localhost:5000`. The backend must be running for the UI to return data.

### Non-obvious gotchas

- There is no system `python`, only `python3`. Backend deps live in a project venv at `.venv`; invoke tools as `.venv/bin/python` / `.venv/bin/pytest` (or activate `.venv`). The venv needs the `python3.12-venv` system package (already baked into the environment).
- `pytest` is **not** listed in `requirements.txt`; the update script installs it into the venv. Run the full suite with `.venv/bin/python -m pytest tests costs/tests automation/tests`.
- No linter/formatter is configured (no eslint/ruff/flake8/prettier). There is no lint step to run.
- API routes are namespaced under `/api/...` with sub-prefixes (e.g. `/api/backtest/assets`, `/api/backtest/run`). There is no bare `/api/assets` endpoint.
- The app pulls live data from external providers (yfinance, ccxt, FRED, etc.), so **outbound internet is required** for backtests, quotes, and most terminal views to return data. No API keys are needed for the basic public-data path; `.env` (copied from `.env.example`) is optional and only enables premium/broker integrations.
- Running the app mutates tracked local-state files under `data/` (e.g. `data/bitcoin_data_cache.json`, `data/intelligence_signals.db`). Do not commit these runtime changes; `git checkout -- data/...` to discard them.
