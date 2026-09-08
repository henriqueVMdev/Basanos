# Graph — Terminal Quantitativo

Terminal de análise e laboratório de estratégias no estilo Bloomberg, construído sobre fontes de dados gratuitas. Backend em Flask (Python), frontend em Vue 3 + Vite + Tailwind, gráficos em Plotly e Lightweight Charts, persistência em SQLite e JSON.

> Documentação completa: [visão geral](docs/README.md), [produto e regras de negócio](docs/PRODUCT.md), [arquitetura](docs/ARCHITECTURE.md), [API](docs/API.md), [operação e produção](docs/OPERATIONS.md) e [diagramas](docs/DIAGRAMS.md).

Princípios do projeto:

- **Fontes gratuitas e rotuladas** — todo dado identifica a origem; o que é proxy ou heurística é marcado como tal na interface.
- **Validação causal** — sinais e backtests usam apenas informação disponível até cada data (expanding windows, sem lookahead). Estratégias só são aprovadas se sobrevivem à validação fora da amostra.
- **Auto-auditoria** — os sinais emitidos são gravados e confrontados com o retorno realizado depois; o app mede a si mesmo, sem viés de retrospectiva.

---

## Como rodar

### Pré-requisitos

- Python 3.11+
- Node.js 18+

### Backend

```bash
pip install -r requirements.txt
python server.py
```

O servidor Flask sobe em `http://localhost:5000` (API em `/api/...`). Nenhuma chave é obrigatória para o uso básico.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

O Vite sobe em `http://localhost:5173` com proxy para a API. Build de produção: `npm run build`.

### Configuração opcional (`.env`)

Copie `.env.example` para `.env` e preencha só o que for usar. O `.env` não vai para o git.

| Variável | Habilita |
|---|---|
| `BYBIT_DEMO_API_KEY/SECRET` | Automação com ordens em conta demo Bybit |
| `BYBIT_REAL_*` / `BYBIT_PROP_*` | Automação em conta real (permissão só de trade) |
| `BINGX_*`, `OKX_*`, `HYPERLIQUID_*` | Sincronização do Diário com as corretoras (chaves read-only) |
| `EIA_API_KEY` | Estoques/produção de energia (CDTY, grátis) |
| `GLASSNODE_API_KEY` | Métricas on-chain BTC premium (há fallback gratuito) |
| `CRYPTOQUANT_*_URL` | Séries proprietárias CryptoQuant (Whale Ratio etc.) |
| `SEC_API_USER_AGENT` | User-Agent identificável exigido pelo EDGAR |

### Testes

```bash
python -m pytest tests/
```

---

## Funcionalidades

### Terminal de mercado

| Seção | Rota | O que faz |
|---|---|---|
| Monitor | `/monitor` | Watchlist com quotes cripto e tradfi, sparklines e atalhos |
| Screener | `/screener` | Varredura de mercado cripto/tradfi por variação, volume e funding |
| EQS | `/eqs` | Screener de ações e fundos por múltiplos fundamentais |
| DES | `/des` | Ficha descritiva do ativo (perfil, múltiplos, dados corporativos) |
| EA | `/ea` | Análise de balanços: receita, margens, estoques por trimestre |
| Rates | `/rates` | Juros, curvas de treasury e spreads de crédito |
| OMON | `/omon` | Cadeias de opções com greeks (Black-Scholes local), superfície de vol e simulador de estratégias |
| Book | `/book` | Livro de ofertas e negócios recentes (cripto) |
| CDTY | `/cdty` | Commodities: curvas de futuros (contango/backwardation), clima nas regiões produtoras com leitura didática de impacto no preço, frete/shipping, estoques EIA |
| ALTD | `/alt` | Dados alternativos: indicadores proprietários, supply chain (GSCPI), tráfego aéreo TSA, clima/ENSO, estoques setoriais, microestrutura cripto (funding de todos os perps), on-chain |
| News | `/news` | Agregador RSS (cripto, mercados, commodities) com busca |
| Sazonalidade | `/seasonality` | Curvas sazonais normalizadas (20/15/10/5/2 anos), estatísticas mensais, heatmap ano a ano e tendência intramês para qualquer ativo |
| Calendário | `/calendar` | Eventos macro e earnings |
| Alertas | `/alerts` | Alertas de preço, funding e score multifator; vigia automático de mudanças de sinal (ver Central de Inteligência) |
| Trade | `/trade` | OMS paper: pre-trade checks, ordens, blotter e TCA |
| Portfolio Lab | `/portfolio-lab` | Análise de carteira: correlações, fronteiras e métricas de risco |

### Central de Inteligência (`/intelligence`)

Motor de cruzamento: cada fonte que o app coleta vira um fator que vota -1/0/+1 sobre o ativo; o rótulo (COMPRA/VENDA/NEUTRO) vem da soma, a confiança é a concordância entre fatores e a cobertura é a fração de fontes que respondeu.

Fatores por classe de ativo:

- **Todos**: tendência (SMA200), momentum 30d, RSI(14), sazonalidade do mês corrente, liquidez líquida do Fed (WALCL − RRP − TGA), regime de risco (HY spread + VIX), drivers (correlação 60d com SPX, DXY, WTI e BTC — educativo).
- **BTC**: modelo on-chain profundo com backtest causal de 8 anos (NUPL, STH-MVRV, STH-SOPR, Pi Cycle + técnica subordinada) — ver `research/analyze_btc_onchain_signals.py`.
- **Cripto**: funding do perp (contrário), TVL da chain, Fear & Greed (contrário), amplitude de funding de todos os perps, top traders da Binance, atividade de desenvolvimento.
- **Ações EUA**: put/call ratio por open interest (contrário nos extremos), IV vs vol realizada, movimento implícito pelo straddle ATM, short interest (extremo = combustível de squeeze), filings de insiders (insight), earnings próximos (insight de risco); aéreas usam o tráfego TSA e armadores o GSCPI.
- **Commodities**: posicionamento especulativo COT/CFTC (contrário nos percentis extremos); agrícolas cruzam ENSO e alertas de clima nas regiões produtoras.

Camadas de verificação:

1. **Ranking do universo** — 25 ativos (cripto, commodities CFTC, índices, FX, ações-tema) ranqueados por score, reconstruído em background a cada 15 minutos.
2. **Divergências** — cruzamentos entre fatores (ex.: alta com especuladores lotados no COT, preço subindo com TVL encolhendo).
3. **Validação histórica** — retorno real 7/30 dias à frente quando cada condição valeu no passado do próprio ativo.
4. **Auto-auditoria forward** — cada sinal do ranking é gravado em SQLite (1 por símbolo/dia, nunca reescrito) e confrontado com o retorno realizado em D+7 e D+30.
5. **Alertas automáticos** — mudanças de rótulo e divergências novas viram alertas disparados (badge + toast).
6. **Paper trading do sinal** — a estratégia `intelligence_signal` segue os rótulos na Automação, transformando o sinal em P&L auditável.

### Laboratório de estratégias

| Seção | Rota | O que faz |
|---|---|---|
| Backtest | `/backtest` | Backtests com custos, slippage e relatórios detalhados |
| Otimizador | `/optimizer` | Grid/walk-forward com validação fora da amostra e exportação |
| Gráfico | `/grafico`, `/tech` | Gráficos com indicadores e replay |
| Regime | `/regime` | Detecção de regime de mercado |
| Prop Challenge | `/prop-challenge` | Simulação de desafios de mesa proprietária (Monte Carlo) |
| Diário | `/journal` | Diário de trades com sincronização via API das corretoras |
| Automação | `/automation` | Deployments paper/real: runner processa candles fechados, ordens com TP/SL server-side, guardrails de perda diária/total, comparação realizado vs backtest |

Estratégias em `strategies/` expõem `signal(df, params)` (contrato validado em `automation/signals.py`). Inclui MM9 pullback, Daily Ensemble TSMOM, Donchian e Intelligence Signal, com equivalentes Pine Script para TradingView quando aplicável.

---

## Fontes de dados

| Fonte | Uso no app | Custo/observação |
|---|---|---|
| Yahoo Finance (yfinance) | Preços históricos e quotes tradfi, fundamentos, opções, short interest, earnings, sazonalidade | Gratuito; IV de opções às vezes vem inválida (o app filtra) |
| Bybit / Binance / OKX (CCXT) | OHLCV cripto, funding, open interest, book, execução da automação | APIs públicas; automação real exige chaves |
| CoinGecko | Perfil de moedas (mcap, supply, ATH, dev, comunidade) | Gratuito, rate limit — pool de requisições reduzido |
| DeFiLlama | TVL por chain e stablecoins | Gratuito |
| bitcoin-data.com | NUPL, STH/LTH Realized Price, STH-MVRV, STH-SOPR (fallback do Glassnode) | Gratuito, 10 req/hora — cache em disco de 12h em `data/bitcoin_data_cache.json` |
| Glassnode / CryptoQuant | Mesmas séries em versão premium + métricas de whales | Opcional, requer chave/URL no `.env` |
| mempool.space / blockchair / blockchain.info | Rede Bitcoin: fees, mempool, dificuldade, hashrate, endereços ativos | Gratuito |
| alternative.me | Fear & Greed Index (série 90d) | Gratuito |
| FRED (St. Louis Fed) | Balanço do Fed, RRP, TGA, HY spread, VIX, dólar — liquidez e regime de risco | Gratuito |
| NY Fed | GSCPI (pressão global de supply chain) | Gratuito; arquivo `.xlsx` é na verdade `.xls` (requer `xlrd`) |
| TSA | Passageiros diários em checkpoints (proxy de demanda aérea) | Gratuito, tabela HTML oficial |
| NOAA CPC | ENSO/ONI (El Niño / La Niña) | Gratuito |
| met.no | Previsão 7 dias nas regiões produtoras de commodities agrícolas | Gratuito, exige User-Agent |
| SEC EDGAR | Filings de insiders (Forms 3/4/5) | Gratuito, exige User-Agent identificável |
| CFTC | Commitments of Traders (posicionamento especulativo, 2 anos) | Gratuito |
| EIA | Estoques/produção/demanda de energia | Gratuito com registro (chave no `.env`) |
| Feeds RSS (CoinDesk, Cointelegraph, MarketWatch, InfoMoney, OilPrice etc.) | Agregador de notícias | Gratuito |

---

## Estrutura do repositório

```
server.py               ponto de entrada Flask (porta 5000)
config.py / loader.py   configuração e leitura de CSVs de otimização

api/                    Blueprints Flask
  terminal_api.py         endpoints do terminal (/api/terminal/...)
  agents_api.py           agentes (CRUD, execução SSE, skills)
  hft_engine.py           motor HFT experimental

providers/              coleta e normalização de dados externos
  market_data.py          OHLCV e exchanges (CCXT)
  intelligence_data.py    motor de cruzamento, ranking, tracking forward
  btc_onchain_metrics.py  métricas on-chain BTC (bitcoin-data/Glassnode, Pi Cycle, OI)
  onchain_data.py         overview on-chain e perfil por moeda
  altdata.py              GSCPI, TSA, clima/ENSO, setores, cripto micro
  commodities_data.py     curvas de futuros, clima nas regiões, shipping, EIA
  seasonality_data.py     análise sazonal por ativo
  insider_data.py         SEC EDGAR, CFTC COT, top traders Binance
  liquidity_data.py       séries FRED
  markets_data.py         opções, book, dados de mercado tradfi
  technical_data.py       indicadores técnicos e gráficos
  tradfi_data.py          ações, índices, FX, renda fixa
  eqs_data.py             dados de equities
  calendar_data.py        calendário econômico
  exchange_sync.py        sincronização de trades/taxas reais (CCXT)
  x_scraper.py            coleta no X/Twitter

engine/                 cálculo e estado de trading
  backtesting.py          motor de backtest
  optimizer.py            otimização e walk-forward
  oms.py                  OMS paper (pre-trade, blotter, TCA)
  regime_detection.py     detecção de regimes de mercado
  portfolio_lab.py        métricas de portfólio
  equity_analysis.py      análise de equities
  options_analytics.py    greeks, superfície de vol, simulador de estratégias

automation/             runner, engine paper, executor Bybit, SQLite
strategies/             estratégias automatizáveis (signal()) e Pine Scripts
costs/                  fees, funding, métricas e cenários
monte_carlo/            simulações e permutation tests
charts/                 gráficos Plotly compartilhados
research/               scripts de pesquisa com validação causal
streamlit_app/          dashboard Streamlit legado (app.py + pages/)
frontend/               Vue 3 + Vite + Tailwind (porta 5173)
data/                   caches, bancos e estado local (sinais, on-chain, journal, alertas)
docs/                   documentação de arquitetura, API e operação
tests/                  testes de regressão
```

---

## Avisos

- Este projeto é ferramenta de estudo e pesquisa quantitativa. Nada aqui é recomendação de investimento.
- Sinais, backtests e paper trading não garantem resultado futuro; sazonalidade histórica não garante repetição.
- O modo real da automação opera dinheiro de verdade na Bybit — use chaves com permissão apenas de trade e entenda os guardrails antes de ativar.
