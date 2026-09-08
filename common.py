"""Helpers compartilhados por providers, engine e api."""

import math
import time


def ttl_cache():
    """Cria um memoizador com TTL por chave: cached(key, ttl_s, fn).

    Store próprio por módulo, evitando colisão de chave entre chamadores.
    Sem lock: corrida concorrente no máximo refaz o fetch.
    """
    store: dict = {}

    def cached(key, ttl_s, fn):
        hit = store.get(key)
        if hit and time.time() - hit[0] < ttl_s:
            return hit[1]
        data = fn()
        store[key] = (time.time(), data)
        return data

    return cached


def to_float(v):
    """float ou None. NaN e infinito viram None: não são JSON válido."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None
