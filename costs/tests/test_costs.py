"""
Testes obrigatórios do módulo de custos (CLAUDE.md). Sem rede: CCXT é mockado.
Inclui a validação de sanidade do trade calculado à mão ($92,75).
"""

from decimal import Decimal

import pytest

from costs.models import Trade, FundingEvent
from costs.config import ExchangeFees, DEFAULT_FEES
from costs.calculator import CostCalculator
from costs.funding import fetch_funding


TAKER = Decimal("0.0005")
FEES = ExchangeFees(maker=Decimal("0.0002"), taker=TAKER)


def make_trade(side="long", qty="0.1", entry="60000", exit="61000",
               entry_time=1000, exit_time=2000):
    return Trade(side=side, qty=Decimal(qty), entry_price=Decimal(entry),
                 exit_price=Decimal(exit), entry_time=entry_time, exit_time=exit_time)


def calc(**kw):
    return CostCalculator("binance", FEES, **kw)


# 1. Sinal do funding
@pytest.mark.parametrize("side,rate,expect_negative", [
    ("long", "0.0001", True),    # long + rate positivo → paga
    ("long", "-0.0001", False),  # long + rate negativo → recebe
    ("short", "0.0001", False),  # short + rate positivo → recebe
    ("short", "-0.0001", True),  # short + rate negativo → paga
])
def test_sinal_funding(side, rate, expect_negative):
    tr = make_trade(side=side)
    ev = [FundingEvent(timestamp=1500, rate=Decimal(rate))]
    b = calc().apply_trade(tr, ev)
    if expect_negative:
        assert b.funding_total < 0
    else:
        assert b.funding_total > 0


# 2. Notional alavancado: fee de 10x = 10× fee sobre a margem
def test_notional_alavancado():
    # margem $100, 10x → notional $1000; qty = 1000/entry
    entry = Decimal("100")
    qty_margin = Decimal("1")        # notional $100 (1x sobre a margem)
    qty_lev = Decimal("10")          # notional $1000 (10x)
    t1 = Trade("long", qty_margin, entry, entry, 1, 2)
    t10 = Trade("long", qty_lev, entry, entry, 1, 2)
    f1 = calc().apply_trade(t1, []).fee_entrada
    f10 = calc().apply_trade(t10, []).fee_entrada
    assert f10 == f1 * 10


# 3. Janela de funding
def test_funding_fora_da_janela_nao_aplica():
    tr = make_trade(entry_time=1000, exit_time=2000)
    ev = [FundingEvent(timestamp=999, rate=Decimal("0.001")),
          FundingEvent(timestamp=2001, rate=Decimal("0.001"))]
    b = calc().apply_trade(tr, ev)
    assert b.n_funding_events == 0
    assert b.funding_total == 0


def test_funding_nas_bordas_aplica():
    tr = make_trade(entry_time=1000, exit_time=2000)
    ev = [FundingEvent(timestamp=1000, rate=Decimal("0.0001")),
          FundingEvent(timestamp=2000, rate=Decimal("0.0001"))]
    b = calc().apply_trade(tr, ev)
    assert b.n_funding_events == 2


# 4. Round trip taker
def test_round_trip_taker():
    tr = make_trade(entry="60000", exit="60000")  # mesmo preço dos dois lados
    b = calc().apply_trade(tr, [])
    notional = Decimal("0.1") * Decimal("60000")
    assert b.fee_entrada == notional * TAKER
    assert b.fee_saida == notional * TAKER
    assert b.fee_entrada + b.fee_saida == 2 * notional * TAKER


# 5. Paginação de funding: 250 eventos, sem duplicata
class FakeExchange:
    def __init__(self, total, step=8 * 3600 * 1000, start=0):
        self.events = [{"timestamp": start + i * step, "fundingRate": 0.0001}
                       for i in range(total)]

    def fetch_funding_rate_history(self, symbol, since=0, limit=100):
        batch = [e for e in self.events if e["timestamp"] >= since][:limit]
        return batch


def test_paginacao_250_eventos():
    ex = FakeExchange(250)
    until = ex.events[-1]["timestamp"] + 1
    res = fetch_funding(ex, "BTC/USDT:USDT", 0, until)
    assert len(res) == 250
    timestamps = [r["timestamp"] for r in res]
    assert len(set(timestamps)) == 250          # sem duplicatas
    assert timestamps == sorted(timestamps)     # ordenado


# 6. Intervalo variável (OKX): usa timestamps reais, não assume 8h
def test_intervalo_variavel_usa_timestamps_reais():
    # Eventos com espaçamento irregular (1h, 4h, 8h) — todos dentro da janela
    tr = make_trade(entry_time=0, exit_time=100_000_000)
    ev = [FundingEvent(timestamp=t, rate=Decimal("0.0001"))
          for t in [1_000_000, 5_000_000, 13_000_000]]
    b = calc().apply_trade(tr, ev)
    # Aplica exatamente os 3 eventos fornecidos, independente do espaçamento
    assert b.n_funding_events == 3


# 7. Determinismo
def test_determinismo():
    tr = make_trade()
    ev = [FundingEvent(timestamp=1500, rate=Decimal("0.0001"))]
    a = calc().apply_trade(tr, ev)
    b = calc().apply_trade(tr, ev)
    assert a == b


# 8. Validação de sanidade ($92,75)
def test_sanidade_manual_92_75():
    tr = make_trade(side="long", qty="0.1", entry="60000", exit="61000",
                    entry_time=1000, exit_time=3000)
    ev = [FundingEvent(timestamp=1500, rate=Decimal("0.0001")),
          FundingEvent(timestamp=2500, rate=Decimal("0.0001"))]
    b = calc().apply_trade(tr, ev)
    assert b.pnl_bruto == Decimal("100.0")
    assert b.fee_entrada + b.fee_saida == Decimal("6.05")
    assert b.funding_total == Decimal("-1.20")
    assert b.pnl_liquido == Decimal("92.75")


# Cenário pessimista é pior que o realista
def test_pessimista_pior_que_realista():
    tr = make_trade()
    ev = [FundingEvent(timestamp=1500, rate=Decimal("0.0001"))]
    realista = calc().apply_trade(tr, ev)
    pessimista = calc(funding_multiplier=Decimal("1.5"),
                      slippage_pct=Decimal("0.0005")).apply_trade(tr, ev)
    assert pessimista.pnl_liquido < realista.pnl_liquido
