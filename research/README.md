# Pesquisa — MM9 Pullback Maker / Prop Challenge

Scripts que geraram os números do `RELATORIO_prop_challenge.md` (raiz do
projeto) e do `strategies/mm9_pullback_maker.yaml`.

## Reprodução

```bash
py research/download_data.py     # 12 perps Bybit: 15m 1 ano + funding -> research/data/
py research/research5_fix.py     # BTC..DOGE: 24h vs janela 19-00h BRT (fix de lookahead)
py research/research6_multi.py   # 12 simbolos na janela + ranking de TREINO
py research/prop_mc_combo.py     # fronteira solo/cesta x risco (MC, forward-only)
py research/combo_low_risk.py    # cestas grandes x risco baixo
```

## Regras de honestidade embutidas (não relaxar)

1. **Sem lookahead**: filtros no candle anterior; slope 1h só com horas
   FECHADAS (bucket disponível em `start + 1h` — nunca `resample+ffill` cru).
2. **Fill maker exige atravessar** a limite (`Low < limite`), não só tocar.
3. **TP e SL no mesmo candle = LOSS.**
4. Custos Bybit reais (maker 0.02% / taker 0.055%) + funding histórico real;
   cenário stress = funding 1.5x + slippage 0.05% nas saídas a mercado.
5. **Seleção de cesta APENAS com dados de TREINO** (70%); o Monte Carlo do
   prop challenge avalia apenas o FORWARD (30%).
6. Monte Carlo por **bootstrap de dias inteiros** (preserva a correlação
   entre trades do mesmo dia/símbolos simultâneos); violação da perda
   diária de -5% reprova.

`research/data/` é gerado e não deve ir para o git.

## Pesquisa diária (TSMOM) — CONCLUÍDA, veredito NEGATIVO (2026-07-08)

Pipeline: `download_daily.py` → `daily_lab.py` (treino <2024) →
`daily_validate.py` (validação 2024-2025, uma passada) → `daily_final.py`
(holdout 2026 jan-jul, intocado até o fim).

Resultado após custos taker+slip+funding, 13 perps Binance:

| família              | treino (<2024) | valid. 24-25 | holdout 2026 |
|----------------------|----------------|--------------|--------------|
| tsmom 21d long-only  | PF 2.97        | PF 1.46      | **PF 0.25** (n=131, exp −2.5%) |
| tsmom 28d L/S        | PF 2.89        | PF 1.36→0.96 | **PF 0.98** (n=277, ~zero)     |

Por ano: o edge era forte 2020-2021, negativo 2022, ok 2023-2024,
**negativo em 2025 e 2026**. No corte por símbolo 2024+, DOGE e XRP
carregam quase todo o lucro — o resto é ruído. Conclusão honesta:
**TSMOM diário em cripto não sobrevive a 2025-2026 depois de custos**;
o prêmio documentado na literatura (Liu & Tsyvinski 2021) decaiu/foi
arbitrado. Sem estratégia diária aprovada.

**O holdout 2026 está queimado**: qualquer nova iteração diária avaliada
nele é overfitting de seleção. Nova pesquisa no diário exige esperar
dados novos (2026 H2+) ou mudar de classe de sinal (ex.: carry de
funding, cross-sectional) com o mesmo protocolo treino/val/holdout.

### Otimizações estruturais (`daily_opt.py`, 2026-07-08)

Protocolo anti-overfit: desenvolvimento 100% no treino; só regras
uniformes da literatura (nada tunado por símbolo); teste de platô nos
vizinhos; validação 2024-25 usada **uma vez** num pacote pré-comprometido.

Pacote: **ensemble TSMOM (7/14/21/28, maioria) + regime BTC>SMA200
(bull=só long, bear=só short) + vol targeting por trade (3%/ATR%, cap
1.5x)**. Platô limpo (MA150-250 e thr 0.25-0.5 ≈ iguais).

| janela            | resultado |
|-------------------|-----------|
| treino <2024      | PF 2.75, sh~2.2, **positivo em todos os anos incl. 2022** (+179 vs −865 do 21d LO) |
| validação 24-25 (1 passada) | PF 1.64, sh~1.27, +1.0%/trade (2024 +730, 2025 −31) |
| 2026 (descritivo) | PF 2.09, +231% (n=165) — regime permitiu shorts no bear |

Ressalva honesta: o 2026 já tinha sido visto (sabíamos que long-only
morreu num ano bear), então o número 2026 é **descritivo, não
aprovação** — o filtro de regime foi motivado por 2022 (treino), mas há
risco de contaminação indireta. Status: **candidata promissora,
aprovação final só com forward/paper trading em dados novos.**

### Forward/paper (2026-07-09)

Pacote em paper via `strategies/daily_ensemble.py` + automação estendida
(ordem `market` com fill na abertura seguinte + `exit_on_flip`
posicional; paridade runner≡sinal do research coberta em
`automation/tests/test_parity_daily.py`). 13 deployments paper 1d
Binance criados por `research/deploy_daily.py` ("DailyEns *").

Premissas do paper: sem slippage e sem funding (o backtest cobrou
0.05%/perna + funding) ⇒ paper é otimista ~0.1%/rodada + funding;
descontar isso na leitura. Só paper — o executor demo/real (Bybit) não
suporta market/posicional ainda.

Entregáveis do pacote (espelham o research, sinal verificado idêntico em
BTC/ETH/SOL/DOGE):
- `strategies/tsmom.py` — backtest `run()` (frontend) + `signal()`
  ao vivo; regime BTC via `market_data`.
- `strategies/tsmom.pine` — TradingView (gráfico 1D); regime via
  `request.security(BTCUSDT, "1D")`.
- `strategies/daily_ensemble.py` — gêmea de automação já em paper.

### DePaula no diário (`depaula_opt.py`, 2026-07-09) — REPROVADA

Núcleo do sinal (ângulo de MA + histerese, matemática validada idêntica
ao `engine/backtesting.py`) rodado no motor honesto (`run_daily`), grid de 576
combos 100% no treino <2024. O motor original **não serve** para
aferir edge: executa no close da mesma barra do sinal, sem custos, e
checa TP/SL contra bandas da MA da própria barra (lookahead); além do
clamp `max(atr, 0.01)` do mintick que distorce moedas de preço baixo.

Treino: platô largo e consistente em EMA lenta (50-200, lb 5-10,
th 0.3-1.0, L/S, exit-on-flat) — PF 7-9, 2020-23 todos positivos.
Pacote pré-comprometido (centro do platô: EMA 100, lb 5, th 0.5, flat,
L/S) na validação 24-25 (1 passada): **PF 1.18, sharpe ~0.39**
(2024 PF 1.47, 2025 PF 0.92). O PF de treino era beta dos mega-trends
2020-21. Veredito: sobrevive aos custos por pouco, mas é **inferior ao
daily ensemble já em paper (PF 1.64, sharpe 1.27)** — não aprovada
para deploy. Grid completo em `research/data/depaula_grid.csv`.
