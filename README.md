# Basanos

Laboratório de pesquisa quantitativa. É um app web para testar hipóteses sobre séries temporais de mercado e descobrir quais delas sobrevivem a uma validação honesta. O backend é Python (Flask, Pandas, NumPy, SciPy, statsmodels) e o frontend é Vue 3 com Vite e Tailwind.

O nome vem do grego *básanos*, a pedra de toque que se usava para saber se um metal era ouro de verdade. A ideia do projeto é parecida: cada estratégia passa por testes que separam resultado real de sorte e overfitting.

> O projeto se chamava Graph durante o desenvolvimento, e a documentação em `docs/` ainda usa esse nome.

## O que ele faz

O caminho principal é pesquisar, otimizar, validar e só depois simular a execução.

- Backtest com fees, slippage e funding reais por exchange, mais um cenário pessimista
- Otimização em grid e walk-forward analysis, separando janelas in-sample e out-of-sample
- Monte Carlo e permutation tests para checar se o resultado se distingue do acaso
- Detecção de regime de mercado
- Execução paper com o mesmo contrato de sinal do backtest; testes de paridade conferem se entrada, saída e custos batem entre os dois
- Painéis de dados de mercado, macro e alternativos, usados como contexto e como fatores de um score explicável

As telas e as regras de cada etapa estão em [docs/PRODUCT.md](docs/PRODUCT.md).

## Princípios

Tudo é causal. Sinais e backtests só enxergam a informação disponível até cada data, com janelas expansivas e candles em formação descartados, porque um backtest com lookahead parece ótimo e não se reproduz depois.

Custo entra na tese desde o começo. Estratégia que só funciona sem fees e funding não passa.

O app também audita a si mesmo: os sinais diários ficam gravados sem reescrita e são confrontados com o retorno realizado em D+7 e D+30.

As fontes de dados são gratuitas e sempre rotuladas. O que é proxy ou heurística aparece marcado como tal na interface.

## Arquitetura

É um monólito modular com dois processos: a SPA Vue servida por Nginx e a API Flask, que também roda os cálculos pesados e o runner de execução. O estado fica local, em SQLite (modo WAL), JSON e Parquet.

```
frontend/      Vue 3, Pinia, Vue Router, Plotly, Lightweight Charts
server.py      app Flask: backtest, otimizador, WFA, Monte Carlo
api/           blueprints do terminal de dados e dos agentes
engine/        backtest, otimizador, OMS paper, regimes, analytics
strategies/    estratégias plugáveis
automation/    runner, engine paper, executor de exchange, SQLite
costs/         fees, funding e cenários de custo
monte_carlo/   simulações e permutation tests
providers/     coleta e normalização de dados externos
research/      scripts de pesquisa com validação temporal
tests/         testes de regressão
```

Cada estratégia é um módulo em `strategies/` que expõe `signal(df, params)`, e o contrato do retorno é validado em `automation/signals.py`. O loader só aceita nomes simples, sem pontos, barras ou `_`, para evitar path traversal.

O projeto assume operação local em um único nó: não há fila, banco servidor nem autenticação. Os trade-offs e os débitos conhecidos estão em [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Como rodar

Precisa de Python 3.11+ e Node 18+.

```bash
pip install -r requirements.txt
python server.py        # API em http://localhost:5000
```

```bash
cd frontend
npm install
npm run dev             # http://localhost:5173, com proxy para a API
```

Com Docker, copie `.env.example` para `.env` e rode `docker compose up --build`. O frontend sobe em http://localhost:8080.

Nenhuma chave de API é obrigatória para o uso básico. As integrações opcionais estão descritas no `.env.example`.

## Testes

```bash
python -m pytest
```

São 56 testes cobrindo motor de backtest, custos, Monte Carlo, otimizador, loader, erros da API e paridade entre backtest e execução. Os 5 testes de paridade com dados históricos são pulados se `research/data` não existir; os scripts de download estão em `research/`.

## Documentação

- [Visão geral](docs/README.md)
- [Produto e regras de negócio](docs/PRODUCT.md)
- [Arquitetura](docs/ARCHITECTURE.md)
- [API](docs/API.md)
- [Operação](docs/OPERATIONS.md)
- [Diagramas](docs/DIAGRAMS.md)

## Aviso

É um projeto de estudo e pesquisa. Nada aqui é recomendação de investimento, e resultado de backtest não garante resultado futuro. O executor da automação também suporta conta real; quem for usar deve criar chaves sem permissão de saque e entender os guardrails antes.
