import time

from common import to_float, ttl_cache


def test_cache_serves_hit_until_ttl_expires():
    calls = []
    cached = ttl_cache()

    def fetch():
        calls.append(1)
        return len(calls)

    assert cached("k", 60, fetch) == 1
    assert cached("k", 60, fetch) == 1
    assert cached("outra", 60, fetch) == 2
    assert cached("k", -1, fetch) == 3
    assert len(calls) == 3


def test_cache_stores_are_isolated_per_instance():
    a, b = ttl_cache(), ttl_cache()
    assert a("k", 60, lambda: "a") == "a"
    assert b("k", 60, lambda: "b") == "b"


def test_cache_key_expires_after_ttl():
    cached = ttl_cache()
    assert cached("k", 0.01, lambda: 1) == 1
    time.sleep(0.02)
    assert cached("k", 0.01, lambda: 2) == 2


def test_to_float_rejects_values_that_break_json():
    assert to_float("1.5") == 1.5
    assert to_float(3) == 3.0
    assert to_float(None) is None
    assert to_float("abc") is None
    assert to_float(float("nan")) is None
    assert to_float(float("inf")) is None
