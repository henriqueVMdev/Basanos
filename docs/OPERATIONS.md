# Operação e produção

## 1. Desenvolvimento local

Pré-requisitos atuais: Python 3.11+ e Node.js 18+; os Dockerfiles usam Python 3.12 e Node 22.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python server.py
```

Em outro terminal:

```bash
cd frontend
npm ci
npm run dev
```

Frontend: `http://localhost:5173`. Backend: `http://localhost:5000`.

## 2. Testes e build

```bash
python -m pytest tests costs/tests automation/tests
cd frontend && npm run build
```

Suítes existentes cobrem regressões de calendário/dados, custos e funding, determinismo e paridade entre backtest e runner. Ainda faltam testes de contrato HTTP, frontend, carga, falhas de terceiros, migrações e execução real sandbox end-to-end.

## 3. Variáveis de ambiente

Use `.env.example` como catálogo. Nenhuma chave é necessária para o caminho básico com dados públicos.

Grupos sensíveis:

- Bybit demo/real/prop: execução automatizada.
- BingX/OKX/Hyperliquid: sincronização do diário.
- Glassnode/CryptoQuant/EIA/SEC: enriquecimento de dados.
- OpenRouter/OpenClaw/Hermes: gateways dos agentes.

Regras:

- Nunca versionar `.env`, private keys ou respostas com segredo.
- Em produção, usar secret manager e rotação, não arquivo compartilhado.
- Chaves de exchange: mínimo privilégio, whitelist de IP quando disponível, saque desabilitado.
- Separar credenciais de demo, pessoal e prop.

## 4. Docker Compose atual

```bash
docker compose up --build
```

- Frontend: `http://localhost:8080`.
- Nginx serve a SPA e encaminha `/api` para o backend.
- Backend expõe `5000` diretamente no host.
- `./data` é montado em `/app/data`.

## 5. Bloqueadores antes de produção

O Compose atual é adequado para avaliação local, não para exposição pública. Corrigir antes de produção:

1. O container executa `app.run(debug=True)`, servidor de desenvolvimento Flask e debugger potencialmente inseguro. Usar Gunicorn/uWSGI com `FLASK_DEBUG=0`.
2. `automation/state.db` e `costs/_cache` não estão no volume `data`; podem ser perdidos em recriação do container.
3. Estado JSON (`journal`, `alertas`, `OMS`, HFT, agentes) vive em `data/`, que é volume no Compose.
4. Não há autenticação/RBAC. Não expor APIs de ordens, automação, agentes ou mutação à internet.
5. A porta 5000 é publicada; em produção deve ficar apenas na rede interna.
6. Não existem health/readiness probes, limites de CPU/memória ou política de backup.
7. Runner em thread e optimizer global não suportam múltiplos workers sem coordenação.
8. Respostas de alguns endpoints incluem traceback; remover em produção.

## 6. Topologia recomendada de primeira produção

Para um único operador e volume moderado:

- TLS e autenticação no reverse proxy/IdP.
- Nginx → Gunicorn com um único worker para manter semântica atual do runner, múltiplas threads controladas.
- Runner em processo separado assim que possível.
- Volume persistente explícito para todos os bancos/JSON/caches necessários.
- Backup diário, retenção e teste de restore.
- Rede de saída restrita e allowlist para APIs necessárias.
- Logs estruturados para stdout, coleta central e alertas.

Para multiusuário ou alta disponibilidade: PostgreSQL, Redis/queue, workers de cálculo, scheduler/runner com lease distribuído, object storage para datasets e autenticação/RBAC nativos.

## 7. Observabilidade mínima

Adicionar:

- `/health/live` para processo e `/health/ready` para DB/dependências críticas.
- Métricas: latência/erros por endpoint, jobs ativos, duração de backtest/WFA, cache hit, rate limit, idade do último candle, skew do relógio, deployments/posições, falhas de reconciliação.
- Logs com timestamp UTC, nível, request/deployment id, símbolo e exchange; nunca segredo.
- Alertas: runner morto com deployment ativo, candle atrasado, posição divergente, guardrail, falha persistente de exchange, disco/DB e backup.

## 8. Backup e recuperação

Inventariar e copiar de forma consistente:

- `automation/state.db` com arquivos WAL/SHM ou via backup SQLite online.
- `data/intelligence_signals.db`.
- `data/agents.json`, `data/hft_state.json`.
- `data/journal_data.json`, `data/alerts_data.json`, `data/oms_data.json`.
- Configurações/estratégias versionadas no Git.

RPO/RTO devem ser definidos pelo uso real. Para dinheiro real, validar restore antes da ativação e reconciliar exchange como fonte de verdade após qualquer incidente.

## 9. Runbooks

### Runner parado

1. Consultar `/api/automation/runner/status` e deployments ativos.
2. Verificar logs, relógio do host, rede e API da exchange.
3. Conferir posições/ordens diretamente na exchange.
4. Evitar iniciar uma segunda instância antes de confirmar que a primeira morreu.
5. Reiniciar; o backend religa o runner se encontrar deployment `running`.

### Divergência entre app e exchange

1. Pausar o deployment sem assumir que a posição foi fechada.
2. Registrar estado da exchange e banco.
3. Cancelar ordens órfãs/fechar posição somente por decisão operacional explícita.
4. Reconciliar P&L, fees e timestamps.
5. Corrigir causa e testar em demo antes de retomar real.

### Dados atrasados ou rate limit

1. Identificar fonte e idade do último dado.
2. Verificar se houve fallback/cache vencido.
3. Reduzir chamadas concorrentes e respeitar janela da fonte.
4. Não promover sinais ou operar com timestamp incerto.

### Banco/JSON corrompido

1. Parar componentes escritores.
2. Preservar cópia do arquivo e WAL.
3. Restaurar último backup válido.
4. Reconciliar estado operacional com exchange.

## 10. Checklist de release

- Testes Python e build frontend aprovados.
- Migração de dados testada em cópia.
- Estratégias/configs versionadas e changelog atualizado.
- Segredos presentes no ambiente correto e ausentes dos logs/imagem.
- Backup e rollback confirmados.
- Endpoints mutáveis protegidos.
- Runner único e posição reconciliada.
- Fontes externas essenciais disponíveis.
- Smoke test: assets, backtest, WFA, criação paper e status.

