import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

// ─── Dashboard API ──────────────────────────────────────────────────────────

export function getFiles() {
  return api.get('/files')
}

export function loadCsvByName(filename, filters = {}) {
  return api.post('/load', { filename, filters })
}

export function loadCsvUpload(file, filters = {}) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('filters', JSON.stringify(filters))
  return api.post('/load', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export function filterData(rows, filters) {
  return api.post('/filter', { rows, filters })
}

export function getStrategy(rank, rows) {
  return api.post('/strategy', { rank, rows })
}

export function filterChart(rows, chartType, chartFilters) {
  return api.post('/filter-chart', { rows, chart_type: chartType, chart_filters: chartFilters })
}

// ─── Backtest API ────────────────────────────────────────────────────────────

export function getAssets() {
  return api.get('/backtest/assets')
}

export function getStrategies() {
  return api.get('/backtest/strategies')
}

export function runBacktestAsset(symbol, symbolLabel, interval, config, strategyFile = 'depaula', exchange = '') {
  return api.post('/backtest/run', {
    data_source: 'asset',
    symbol,
    symbol_label: symbolLabel,
    interval,
    config,
    strategy_file: strategyFile,
    exchange: exchange || undefined,
  }, { timeout: 300000 })
}

export function runBacktestCsv(file, config, strategyFile = 'depaula') {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('config', JSON.stringify(config))
  formData.append('strategy_file', strategyFile)
  return api.post('/backtest/run', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export function getCorrelation(tickers) {
  return api.post('/backtest/correlation', { tickers })
}

export function runMonteCarlo(equityCurve, numSims, horizon, seed) {
  return api.post('/backtest/montecarlo', {
    equity_curve: equityCurve,
    num_sims: numSims,
    horizon,
    seed,
  })
}

export function runValidation(payload) {
  return api.post('/backtest/validate', payload, { timeout: 180000 })
}

export function runWfa(payload) {
  return api.post('/backtest/wfa', payload, { timeout: 180000 })
}

export function runCosts(payload) {
  return api.post('/backtest/costs', payload, { timeout: 180000 })
}

export function getChartData(payload) {
  return api.post('/backtest/chart-data', payload, { timeout: 180000 })
}

// Optimizer API
export function getOptimizerGrids(strategyFile = 'depaula') {
  return api.get('/optimizer/grids', { params: { strategy: strategyFile } })
}

export function getOptimizerCount(grid, capital, strategyFile = 'depaula', config = {}) {
  return api.post('/optimizer/count', { grid, capital, strategy_file: strategyFile, config })
}

export function stopOptimizer() {
  return api.post('/optimizer/stop')
}

export function getOptimizerProgress() {
  return api.get('/optimizer/progress')
}

export function startOptimizer(payload) {
  return api.post('/optimizer/run', payload)
}

export function startOptimizerCsv(file, grid, capital, minTrades, rankBy, topN, strategyFile = 'depaula', cycleLongMonths = null, cycleShortMonths = null, config = null) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('grid', JSON.stringify(grid))
  formData.append('capital', capital)
  formData.append('min_trades', minTrades)
  formData.append('rank_by', rankBy)
  formData.append('top_n', topN)
  formData.append('strategy_file', strategyFile)
  if (cycleLongMonths) formData.append('cycle_long_months', JSON.stringify(cycleLongMonths))
  if (cycleShortMonths) formData.append('cycle_short_months', JSON.stringify(cycleShortMonths))
  if (config) formData.append('config', JSON.stringify(config))
  return api.post('/optimizer/run-csv', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export function getOptimizerResult() {
  return api.get('/optimizer/result')
}

// Regime Detection API
export function detectRegimeAsset(symbol, interval, params = {}, exchange = '') {
  return api.post('/regime/detect', {
    source: 'asset',
    symbol,
    interval,
    exchange: exchange || undefined,
    params,
  }, { timeout: 180000 })
}

export function detectRegimeCsv(file, params = {}) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('params', JSON.stringify(params))
  return api.post('/regime/detect', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 180000,
  })
}

// ─── Trade Journal API ────────────────────────────────────────────────────────

export function getJournal() {
  return api.get('/journal')
}

export function setJournalCapital(capitalInicial) {
  return api.post('/journal/capital', { capital_inicial: capitalInicial })
}

export function addTrade(trade) {
  return api.post('/journal/trade', trade)
}

export function updateTrade(id, patch) {
  return api.put(`/journal/trade/${id}`, patch)
}

export function deleteTrade(id) {
  return api.delete(`/journal/trade/${id}`)
}

export function syncJournal(payload = {}) {
  return api.post('/journal/sync', payload, { timeout: 120000 })
}

// ─── Degen (memecoins on-chain) API ──────────────────────────────────────────

export function getDegenChains() {
  return api.get('/degen/chains')
}

export function getDegenTokens(chain, kind = 'trending') {
  return api.get('/degen/tokens', { params: { chain, kind } })
}

export function getDegenHype(chain, token, symbol) {
  return api.get('/degen/hype', { params: { chain, token, symbol }, timeout: 60000 })
}

// Prop Challenge API
export function runPropChallenge(payload) {
  return api.post('/prop-challenge/simulate', payload, { timeout: 300000 })
}

// ─── Automation API ──────────────────────────────────────────────────────────

export function getDeployments() {
  return api.get('/automation/deployments')
}

export function createDeployment(payload) {
  return api.post('/automation/deployments', payload)
}

export function startDeployment(id) {
  return api.post(`/automation/deployments/${id}/start`)
}

export function stopDeployment(id, closePosition = false) {
  return api.post(`/automation/deployments/${id}/stop`, { close_position: closePosition })
}

export function deleteDeployment(id) {
  return api.delete(`/automation/deployments/${id}`)
}

export function getDeploymentStatus(id) {
  return api.get(`/automation/deployments/${id}/status`)
}

export function getRunnerStatus() {
  return api.get('/automation/runner/status')
}

export function getAutomationAccounts() {
  return api.get('/automation/accounts')
}

// ─── Terminal (Bloomberg-like) ───────────────────────────────────────────────

export function getWatch(symbols, exchange = 'bybit', tradfi = []) {
  return api.get('/terminal/watch', {
    params: { symbols: symbols.join(','), exchange, tradfi: tradfi.join(',') },
    timeout: 60000,
  })
}

export function getSpark(symbol, exchange = 'bybit', tf = '15m', bars = 96, market = 'crypto') {
  return api.get('/terminal/spark', { params: { symbol, exchange, tf, bars, market } })
}

export function getScreener(top = 50, exchange = 'bybit', market = 'crypto') {
  return api.get('/terminal/screener', { params: { top, exchange, market }, timeout: 180000 })
}

export function getDes(symbol, exchange = 'bybit', market = 'auto') {
  return api.get('/terminal/des', { params: { symbol, exchange, market }, timeout: 60000 })
}

export function getRates() {
  return api.get('/terminal/rates', { timeout: 120000 })
}

export function getOptionsChain(symbol, expiry) {
  return api.get('/terminal/options', { params: { symbol, expiry }, timeout: 90000 })
}

export function getBook(symbol, exchange = 'bybit', market = 'crypto') {
  return api.get('/terminal/book', { params: { symbol, exchange, market }, timeout: 30000 })
}

export function getEa(symbol) {
  return api.get('/terminal/ea', { params: { symbol }, timeout: 180000 })
}

export function getCdtyOverview() {
  return api.get('/terminal/cdty/overview', { timeout: 120000 })
}

export function getCdtyCurves() {
  return api.get('/terminal/cdty/curves')
}

export function getCdtyCurve(c) {
  return api.get('/terminal/cdty/curve', { params: { c }, timeout: 120000 })
}

export function getCdtyWeather() {
  return api.get('/terminal/cdty/weather', { timeout: 120000 })
}

export function getCdtyShipping() {
  return api.get('/terminal/cdty/shipping', { timeout: 120000 })
}

export function getCdtyInventories() {
  return api.get('/terminal/cdty/inventories', { timeout: 60000 })
}

export function getVolSurface(symbol) {
  return api.get('/terminal/options/surface', { params: { symbol }, timeout: 300000 })
}

export function evalStrategy(payload) {
  return api.post('/terminal/options/strategy', payload, { timeout: 30000 })
}

export function postChart(payload) {
  return api.post('/terminal/chart', payload, { timeout: 180000 })
}

// ── ALTD — dados alternativos ────────────────────────────────────────────

export function getAltIndicators() {
  return api.get('/terminal/alt/indicators', { timeout: 180000 })
}

export function getAltSupplyChain() {
  return api.get('/terminal/alt/supplychain', { timeout: 120000 })
}

export function getAltTraffic() {
  return api.get('/terminal/alt/traffic', { timeout: 60000 })
}

export function getAltClimate() {
  return api.get('/terminal/alt/climate', { timeout: 120000 })
}

export function getAltSectors(s) {
  return api.get('/terminal/alt/sectors', { params: { s }, timeout: 180000 })
}

export function getAltCryptoMicro() {
  return api.get('/terminal/alt/cryptomicro', { timeout: 120000 })
}

export function getAltOnchain() {
  return api.get('/terminal/alt/onchain', { timeout: 120000 })
}

export function getAltOnchainCoin(symbol) {
  return api.get('/terminal/alt/onchain/coin', { params: { symbol }, timeout: 120000 })
}

// ── OMS/EMS — execução de ordens ─────────────────────────────────────────

export function getOmsAccounts() {
  return api.get('/terminal/oms/accounts')
}

export function postPreTrade(payload) {
  return api.post('/terminal/oms/pretrade', payload, { timeout: 60000 })
}

export function submitOmsOrder(payload) {
  return api.post('/terminal/oms/orders', payload, { timeout: 60000 })
}

export function cancelOmsOrder(id) {
  return api.delete(`/terminal/oms/orders/${id}`)
}

export function getOmsBlotter(account = 'paper') {
  return api.get('/terminal/oms/blotter', { params: { account }, timeout: 90000 })
}

export function getOmsTca(account = 'paper') {
  return api.get('/terminal/oms/tca', { params: { account }, timeout: 30000 })
}

export function resetOmsPaper() {
  return api.post('/terminal/oms/reset', { confirm: true })
}

export function getEqsMeta() {
  return api.get('/terminal/eqs/meta')
}

export function runEqsScreen(filters) {
  return api.post('/terminal/eqs/equity', filters, { timeout: 120000 })
}

export function getEqsFunds(screen) {
  return api.get('/terminal/eqs/funds', { params: { screen }, timeout: 120000 })
}

export function getAlerts() {
  return api.get('/terminal/alerts')
}

export function createAlert(payload) {
  return api.post('/terminal/alerts', payload)
}

export function deleteAlert(id) {
  return api.delete(`/terminal/alerts/${id}`)
}

export function getNews(cat = 'all', q = '') {
  return api.get('/terminal/news', { params: { cat, q }, timeout: 60000 })
}

export function getSeasonality(symbol) {
  return api.get('/terminal/seasonality', { params: { symbol }, timeout: 180000 })
}
export function getIntelligence(symbol='BTC') {
  return api.get('/terminal/intelligence', { params:{symbol}, timeout:180000 })
}
export const getIntelligenceRanking=()=>api.get('/terminal/intelligence/ranking',{timeout:60000})
export const getIntelligenceTracking=()=>api.get('/terminal/intelligence/tracking',{timeout:120000})
export const getMarketCalendar=(symbols='')=>api.get('/terminal/calendar',{params:{symbols},timeout:120000})
export const getLiquidity=()=>api.get('/terminal/liquidity',{timeout:120000})
export const runPortfolioLab=(symbols,years=3)=>api.post('/terminal/portfolio-lab',{symbols,years},{timeout:180000})

export default api
