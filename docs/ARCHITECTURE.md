# Arquitetura técnica

## 1. Visão geral

O Graph é um monólito modular com dois processos implantáveis:

- `frontend`: SPA Vue compilada e servida por Nginx.
- `backend`: aplicação Flask que também executa cálculos intensivos e inicia o runner de automação em thread.

Não há fila, scheduler externo, banco servidor ou camada de autenticação. Essa simplicidade favorece desenvolvimento local, mas limita escala horizontal e isolamento em produção.

## 2. Stack

| Camada | Tecnologias | Responsabilidade |
|---|---|---|
| UI | Vue 3, Composition API, Pinia, Vue Router | Telas, estado e navegação |
| Build/UI | Vite, Tailwind CSS | Bundle e design system utilitário |
| Gráficos | Plotly.js, Lightweight Charts | Analytics e candles |
| HTTP | Axios, Flask, Flask-CORS | Cliente REST e API JSON/multipart |
| Quant | Python, Pandas, NumPy, SciPy, statsmodels, scikit-learn | Backtest, otimização, estatística e sinais |
| Mercado | yfinance, CCXT | TradFi e exchanges cripto |
| Persistência | SQLite WAL, JSON, Parquet | Estado operacional, tracking e caches |
| Deploy | Docker, Docker Compose, Nginx | Containers e proxy `/api` |
| Testes | pytest | Custos, regressões e paridade |

## 3. Organização do código

```text
frontend/src/
  views/                 páginas por rota
  components/            componentes de domínio e layout
  stores/                estado Pinia e orquestração de chamadas
  api/client.js          adapter HTTP único
  router/index.js        rotas da SPA

server.py                app Flask e domínios de backtest/optimizer/journal/degen
api/                     Blueprints Flask (terminal, agentes, HFT)
providers/               coleta de dados externos (mercado, on-chain, alternativos)
engine/                  backtest, optimizer, OMS, regimes, analytics
automation/              API, runner, executores, sinais e SQLite
strategies/              plugins de estratégia e Pine Scripts
costs/                   fees, funding, métricas e cenários
monte_carlo/             simulações e permutation tests
charts/                  gráficos Plotly compartilhados
research/                experimentos e validações offline
streamlit_app/           dashboard Streamlit legado
data/                    caches/estado local
tests/                   regressões transversais
```

## 4. Frontend

### Fluxo de estado

`View → Store Pinia → api/client.js → Flask → Store → Componentes`

O `workspace` compartilha seleção de estratégia, símbolo, label, timeframe, exchange e parâmetros entre telas. Stores de domínio mantêm loading, erros, configuração e resultados. Isso simplifica a passagem Backtest/Otimizador/Automação, mas requer cuidado para não conservar parâmetros incompatíveis entre estratégias.

### Rotas

As rotas estão agrupadas conceitualmente em Estratégia, Mercados, Análise e Execução. O bundle faz lazy load das telas do terminal; telas centrais de estratégia são importadas diretamente. Plotly é separado em chunk próprio.

### Comunicação

O Axios usa `baseURL=/api`; no desenvolvimento o Vite encaminha para `localhost:5000`, e em container o Nginx encaminha para `backend:5000`. Timeouts variam de 120 a 300 segundos para cálculos longos.

## 5. Backend

### Módulos HTTP

- `server.py`: dashboard, backtest, custos, WFA, Monte Carlo, optimizer, prop challenge, regimes, journal e degen.
- `api/terminal_api.py`: monitor, screener, rates, options, dados alternativos, OMS, commodities, notícias, inteligência e portfólio.
- `automation/api.py`: ciclo de vida de deployments e status do runner.
- `api/agents_api.py`: CRUD, execução por SSE, skills e propostas.
- `api/hft_engine.py`: status, start, stop, configuração e reset.

### Estratégias plugáveis

O loader aceita somente nomes simples e rejeita `_`, barras, contrabarras e pontos, reduzindo path traversal. Cada módulo pode expor:

- `NAME`, `DESCRIPTION` e schema de parâmetros para descoberta/UI.
- `run(df, params)` para backtest próprio.
- `prepare_optimizer_params(params)` para adaptação do grid.
- `signal(df, params)` para automação.

O módulo é carregado dinamicamente de `strategies/<nome>.py`. A automação mantém cache de módulos em processo.

### Dados de mercado

`_download_data_safe` normaliza todas as fontes para DataFrame com índice temporal e colunas `Open/High/Low/Close/Volume`.

- CCXT: Binance, Bybit, OKX e Hyperliquid; paginação por timeframe e cache em memória de 30 minutos.
- yfinance: fallback/default; 15m/30m limitados a 60 dias, 1h a dois anos; 2h/4h são resample de 1h.
- Em falha da exchange, cache expirado pode ser reutilizado.
- Funding tem cache Parquet no módulo `costs`.

### Cálculo e concorrência

Endpoints quantitativos são síncronos e rodam dentro de workers/threads Flask. O otimizador mantém estado global de progresso e stop; portanto, não é seguro assumir coordenação correta entre múltiplos processos. O runner de automação é singleton por processo e religa quando encontra deployments `running`.

Para produção escalável, cálculos e runner devem migrar para jobs/worker dedicados com coordenação persistente.

## 6. Persistência

| Artefato | Local atual | Uso |
|---|---|---|
| `automation/state.db` | Diretório `automation/` | Deployments, ordens, posições, equity e eventos |
| `data/intelligence_signals.db` | `data/` | Tracking forward dos sinais |
| `data/agents.json` | `data/` | Configuração dos agentes |
| `data/hft_state.json` | `data/` | Estado do motor HFT |
| `data/journal_data.json` | `data/` | Diário de trades |
| `data/alerts_data.json` | `data/` | Alertas |
| `data/oms_data.json` | `data/` | Estado do OMS paper |
| `costs/_cache/*.parquet` | `costs/_cache/` | Funding por exchange/símbolo |
| cache OHLCV | memória do processo | Redução de rate limit |

SQLite da automação usa WAL, conexão por chamada e índices por deployment. JSONs usam persistência simples; alguns contam com lock em memória, mas não foram projetados para múltiplos processos.

## 7. Modelo da automação

Entidades principais:

- `deployments`: configuração, modo, conta, guardrails, status e equity.
- `orders`: entrada/fechamento, side, tipo, validade e status.
- `positions`: fill, TP/SL, tempo, P&L e fees.
- `equity_snapshots`: curva por candle.
- `events`: log de domínio e erros.

O runner baixa candles, remove candles ainda abertos, processa apenas timestamps novos, reconcilia ordens/posições e calcula o próximo sinal. Em paper usa `engine_paper`; demo e real passam pelo executor Bybit.

## 8. Integrações externas

Principais famílias: yfinance/Yahoo; CCXT/exchanges; FRED; SEC EDGAR; EIA; CFTC; Glassnode/CryptoQuant/bitcoin-data; CoinGecko; DeFiLlama; mempool.space; Alternative.me; NOAA; met.no; RSS e gateways de agentes.

Cada integração deve ser tratada como não confiável: timeout, rate limit, schema variável e indisponibilidade são estados normais. O domínio já aplica caches e fallbacks em vários pontos, porém não há circuit breaker uniforme.

## 9. Decisões e trade-offs

| Decisão | Benefício | Custo/risco |
|---|---|---|
| Monólito Flask | Iteração e execução local simples | Arquivo central grande e escala limitada |
| Estratégias Python dinâmicas | Extensão rápida | Plugin roda com privilégios do backend |
| Fontes gratuitas | Baixo custo de pesquisa | SLA, qualidade e rate limits variáveis |
| SQLite/JSON local | Zero infraestrutura externa | Multi-instância e backup mais difíceis |
| Runner em thread do web server | Deploy simples | Acoplamento e risco de duplicação |
| OHLCV no processo | Latência/rate limit menores | Cache perdido no restart e não compartilhado |
| SPA + REST | Separação clara de UI/API | Sem contrato OpenAPI gerado |

## 10. Débitos arquiteturais prioritários

1. Separar servidor WSGI, worker quantitativo e runner de execução.
2. Centralizar persistência em PostgreSQL/Redis ou declarar operação single-node como requisito.
3. Adicionar autenticação, RBAC, rate limiting, CSRF strategy e auditoria imutável.
4. Definir contratos com OpenAPI e validação de payloads.
5. Retirar stack traces das respostas HTTP e padronizar erros/correlation IDs.
6. Instrumentar métricas, logs estruturados, tracing e health/readiness endpoints.
7. Versionar estratégias e snapshots de configuração usados em cada resultado/deployment.
8. Isolar plugins/estratégias não confiáveis.

