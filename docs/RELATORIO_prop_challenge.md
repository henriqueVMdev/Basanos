# Relatório — Estratégia para Prop Challenge (MM9 Pullback Maker)

**Data:** 2026-07-02 · **Dados:** 1 ano (jul/2025→jul/2026), 35.000 candles
15m por ativo, 12 perps USDT da Bybit, fees reais + funding histórico real.
**Reprodução:** `research/README.md`. **Estratégia:** `strategies/mm9_pullback.py`
(app), `strategies/mm9_pullback_maker.pine` (TradingView),
`strategies/mm9_pullback_maker.yaml` (spec).

---

## 1. A estratégia

Pullback à EMA9 no 15m com **entrada por ordem limite (maker)** — long e short:

- **Entrada**: limite no valor da EMA9 do candle anterior; fill exige o preço
  atravessar; validade de 1 candle (reposicionada a cada setup).
- **Filtros** (candle anterior): EMA50>EMA200 e Close>EMA200 (15m); slope da
  EMA9 do 1h a favor (**só horas fechadas**); Close>EMA9; rank de vol
  (ATR14% na janela de 4 dias) ≥ 0,5.
- **Saídas**: TP +0,5% (limite/maker), SL −1,5% (mercado/taker), time-stop 12h.
  TP+SL no mesmo candle = LOSS.
- **Janela de horário**: entradas só 19h–00h Brasília (22:00–03:00 UTC).
  Saídas 24/7 (ordens descansam na exchange). A janela MELHORA o edge
  (BTC: PF 1,16 no 24h → 1,66 na janela).

**Por que maker importa:** com taker nas duas pontas (0,11% ida-volta) a
estratégia morre (PF ~1,0). O MM9 cross clássico a mercado no 15m tem WR
73% quando o acaso daria 75% — não existe edge ali.

## 2. Descobertas de metodologia (custaram caro, não repetir)

1. **Lookahead no filtro 1h**: `resample('1h').last()+reindex(ffill)` deixa o
   candle 15m das 10:00 ver o close das 10:45. Inflava o resultado ~40%.
   Fix: valor do bucket 1h disponível apenas em `start + 1h`.
2. **Pandas 3 usa resolução µs**: `index.astype(int64)//10**6` NÃO dá ms
   (zerava o funding silenciosamente). Usar
   `(index - Timestamp(0)).total_seconds()*1000`.
3. **Seleção olhando o forward é autoengano**: cesta/ativo escolhidos apenas
   com TREINO; o forward só avalia.

## 3. Resultados por ativo (forward 3,5 meses, janela 19–00h, stress*)

| Ativo | n | WR | PF | Exp/trade | Veredito |
|---|---|---|---|---|---|
| BTC | 81 | 81,5% | 1,59 | +0,139% | ✅ |
| ETH | 90 | 82,2% | 1,42 | +0,111% | ✅ |
| SOL | 96 | 80,2% | 1,34 | +0,093% | ✅ |
| DOGE | 94 | 85,1% | 1,72 | +0,163% | ✅ |
| SUI | 98 | 85,7% | 1,70 | +0,162% | ✅ |
| NEAR | 109 | 85,3% | 1,76 | +0,169% | ✅ |
| LINK | 83 | 84,3% | 1,64 | +0,150% | ✅ |
| ADA | 91 | 82,4% | 1,45 | +0,118% | ✅ |
| XRP | 80 | 76,2% | 0,98 | −0,008% | ❌ morreu |
| AVAX | 88 | 77,3% | 0,98 | −0,009% | ❌ morreu |
| BNB | 59 | 72,9% | 0,95 | −0,018% | ❌ morreu |
| LTC | 62 | 69,4% | 0,86 | −0,050% | ❌ morreu |

\* stress = funding 1,5× + slippage 0,05% nas saídas a mercado + fill
exigindo atravessar a limite.

**8 de 12 positivos** — o edge da estratégia é real e transversal, mas:

## 4. A lição central: correlação treino→forward = 0,08 (zero)

Todas as 12 são lucrativas no treino; o **ranking de treino não prevê** quais
morrem no forward (o 1º do treino, XRP, morreu). Consequências medidas:

- **Solo "agressivo" às cegas** (você não sabe o vencedor): aprovação média
  79–84%, **pior caso 44–51%** — loteria.
- **Diversificação absorve o erro de seleção**: a cesta top8 do treino contém
  XRP e AVAX (mortos) e ainda aprova 96%.

## 5. Prop challenge — fronteira honesta

Regras: alvo +5%, perda máx −10%, perda diária −5% (violação = reprova).
Monte Carlo 10.000 sims, bootstrap de dias inteiros, **apenas trades
forward**, pool stress, sizing por risco fixo (SL cheio = risco%).

| Opção | Risco/trade | Aprovação | Mediana | Pior dia hist. |
|---|---|---|---|---|
| 🥇 **Cesta top8 do treino**¹ | **0,50%** | **96,1%** | **12 dias** | −4,5% |
| Cesta top5 do treino | 0,60% | 96,5% | 14 dias | −3,7% |
| Cesta top8 (mesa com regra diária 4%) | 0,40% | 97,3% | ~16 dias | −3,6% |
| **BTC solo (pré-registrado)** | 1,00% | 98,6% | 29 dias | −1,9% |
| BTC solo | 1,50% | 98,9% | 18 dias | — |
| Solo agressivo às cegas (2–3%) | — | ~79–84% méd. | 6–12 dias | pior caso 44–51% |
| Cesta top12 (grande demais) | 1,00% | 71,9% | 3 dias | viola diária |

¹ XRP+SOL+SUI+ADA+LINK+AVAX+NEAR+ETH (ordem do treino). ~8 trades/dia na
janela — exige execução automatizada.

**Recomendação:** cesta top8 @ 0,5%/trade (equilíbrio) ou BTC solo @ 1%
(máxima segurança). Não jogar solo agressivo.

## 6. Condições de invalidação (parar se ocorrer)

- WR real em demo < 72% após 2–4 semanas (o fill maker não sobreviveu à fila);
- fills sistematicamente piores que o backtest (seleção adversa);
- 3 dias seguidos batendo no limite diário.

**Antes de pagar qualquer challenge: 2–4 semanas em demo/testnet Bybit
comparando os fills com o backtest.**

## 7. Ressalvas permanentes

1. **Fill maker é a hipótese mais frágil** de toda a validação (assumimos
   fill quando o preço atravessa a limite; a fila real pode excluir os
   melhores trades).
2. 1 ano de dados, 1 regime de mercado; forward de 3,5 meses (~80–110
   trades/ativo → WR ±8pp de incerteza).
3. Grid de parâmetros foi explorado no treino — risco residual de overfit,
   mitigado pela consistência entre 8 ativos independentes e pelo stress.
