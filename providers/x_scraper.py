"""
Scraper do X (Twitter) via navegador automatizado (Playwright).

Requer uma sessão logada (use uma conta DESCARTÁVEL — automação viola os ToS do X):

    .venv/bin/python -m providers.x_scraper login    # abre navegador e faça login
    .venv/bin/python -m providers.x_scraper search '$CASHCAT'

A sessão fica em data/.x_session.json (no .gitignore — não commitar).
"""

import json
import re
import sys
from pathlib import Path

SESSION_FILE = Path(__file__).parents[1] / "data" / ".x_session.json"
SEARCH_URL = "https://x.com/search?q={query}&src=typed_query&f=live"


def has_session():
    return SESSION_FILE.exists()


def login():
    """Abre navegador visível para login manual; salva cookies ao fechar."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto("https://x.com/login")
        print("Faça login na janela do navegador.")
        print("Quando o feed carregar, volte aqui e aperte ENTER para salvar a sessão.")
        input()
        ctx.storage_state(path=str(SESSION_FILE))
        browser.close()
        print(f"Sessão salva em {SESSION_FILE}")


def _parse_count(s):
    """'1,234' | '12.5K' | '1.2M' -> int"""
    if not s:
        return 0
    s = s.strip().replace(",", "")
    mult = 1
    if s[-1:].upper() == "K":
        mult, s = 1_000, s[:-1]
    elif s[-1:].upper() == "M":
        mult, s = 1_000_000, s[:-1]
    try:
        return int(float(s) * mult)
    except ValueError:
        return 0


def search(query, limit=20, timeout_ms=25000):
    """
    Busca tweets recentes ($SYMBOL) na aba "Latest". Retorna lista de dicts
    no mesmo formato do resto do pipeline de hype, ou None se falhar.
    """
    if not has_session():
        return None
    from playwright.sync_api import sync_playwright
    from urllib.parse import quote

    tweets = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                storage_state=str(SESSION_FILE),
                viewport={"width": 1280, "height": 2000},
                user_agent=("Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
                            "Gecko/20100101 Firefox/128.0"),
            )
            page = ctx.new_page()
            page.goto(SEARCH_URL.format(query=quote(query)),
                      wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_selector('article[data-testid="tweet"]',
                                       timeout=timeout_ms)
            except Exception:
                browser.close()
                return None  # sem resultados ou sessão expirada

            # Um scroll para carregar mais alguns
            page.mouse.wheel(0, 3000)
            page.wait_for_timeout(1500)

            for art in page.query_selector_all('article[data-testid="tweet"]')[:limit]:
                el_text = art.query_selector('[data-testid="tweetText"]')
                text = el_text.inner_text().strip() if el_text else ""
                if not text:
                    continue

                el_user = art.query_selector('[data-testid="User-Name"] a[href^="/"]')
                user = None
                url = None
                if el_user:
                    user = (el_user.get_attribute("href") or "").lstrip("/") or None
                el_time = art.query_selector("time")
                created = el_time.get_attribute("datetime") if el_time else None
                el_link = art.query_selector('a[href*="/status/"]')
                if el_link:
                    url = "https://x.com" + el_link.get_attribute("href")

                metrics = {"reply": 0, "retweet": 0, "like": 0}
                for kind in metrics:
                    el = art.query_selector(f'[data-testid="{kind}"]')
                    if el:
                        metrics[kind] = _parse_count(el.inner_text())
                views = None
                el_views = art.query_selector('a[href$="/analytics"]')
                if el_views:
                    views = _parse_count(el_views.inner_text())

                tweets.append({
                    "text": text,
                    "user": user,
                    "followers": None,
                    "likes": metrics["like"],
                    "retweets": metrics["retweet"],
                    "replies": metrics["reply"],
                    "views": views,
                    "created_at": (created or "")[:16].replace("T", " "),
                    "url": url,
                })
            browser.close()
    except Exception:
        return None
    return tweets if tweets else None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "login"
    if cmd == "login":
        login()
    elif cmd == "search":
        q = sys.argv[2] if len(sys.argv) > 2 else "$DOGE"
        result = search(q)
        if result is None:
            print("Falhou (sem sessão, sessão expirada ou sem resultados).")
            sys.exit(1)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(__doc__)
