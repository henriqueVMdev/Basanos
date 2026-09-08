import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import {
  getAssets,
  getStrategies,
  getOptimizerGrids,
  getOptimizerCount,
  startOptimizer,
  startOptimizerCsv,
  stopOptimizer,
  getOptimizerProgress,
  getOptimizerResult,
} from '@/api/client.js'
import { useBacktestStore } from '@/stores/backtest.js'
import { useWorkspaceStore } from '@/stores/workspace.js'

export const useOptimizerStore = defineStore('optimizer', () => {
  // Seleção compartilhada entre páginas (ativo/timeframe)
  const ws = useWorkspaceStore()
  const _shared = (key) => computed({ get: () => ws[key], set: (v) => { ws[key] = v } })

  // Assets
  const assets = ref({})

  // Estrategias
  const strategies = ref([])
  const selectedStrategy = ref(null)

  // Grids disponiveis do backend (por estrategia)
  const availableGrids = ref({})

  // Configuracao
  const dataSource = ref('asset')
  const selectedCategory = ref('')
  const selectedAssetLabel = _shared('symbolLabel')
  const selectedSymbol = _shared('symbol')
  const interval = _shared('interval')
  const csvFile = ref(null)
  const startDate = ref('')
  const endDate = ref('')

  // Grid
  const gridMode = ref('rapido')
  const customGrid = ref({})
  const useCustomGrid = ref(false)

  // Ciclo sazonal (params fixos, nao variam no grid)
  const cycleLongMonths = ref([1,2,3,4,5,6,7,8,9,10,11,12])
  const cycleShortMonths = ref([1,2,3,4,5,6,7,8,9,10,11,12])

  // Backtest config
  const capital = ref(1000)
  const minTrades = ref(5)
  const rankBy = ref('Score')
  const topN = ref(20)

  // Flag para sinalizar que o grid foi carregado externamente
  const gridLoadedExternally = ref(false)

  // Estado
  const isRunning = ref(false)
  const comboCount = ref(null)
  const comboCountLoading = ref(false)
  const comboCountError = ref(null)
  const comboCountExact = ref(true)
  const comboCountTooMany = ref(false)
  const comboCountLimit = ref(null)
  const results = ref(null)
  const error = ref(null)
  const progress = ref({ current: 0, total: 0, valid: 0 })
  let _progressTimer = null
  let _countRequestId = 0

  // Grid ativo (custom ou preset)
  const activeGrid = computed(() => {
    if (useCustomGrid.value) return customGrid.value
    const preset = availableGrids.value[gridMode.value]
    if (!preset) return customGrid.value

    const schema = selectedStrategy.value?.schema || []
    const typeMap = {}
    for (const section of schema) {
      for (const field of (section.fields || [])) {
        typeMap[field.key] = field.type
      }
    }

    const typed = {}
    for (const [key, vals] of Object.entries(preset)) {
      const ft = typeMap[key]
      if (ft === 'checkbox') {
        typed[key] = vals.map(v => v === 'True' || v === 'true' || v === true)
      } else if (ft === 'number') {
        typed[key] = vals.map(v => parseFloat(v))
      } else {
        typed[key] = vals.map(v => String(v))
      }
    }
    return typed
  })

  const strategyFile = computed(() => selectedStrategy.value?.file || 'depaula')

  // Actions
  async function fetchAssets() {
    try {
      const { data } = await getAssets()
      assets.value = data.assets || {}
      const categories = Object.keys(assets.value)
      if (categories.length > 0 && !selectedCategory.value) {
        selectedCategory.value = categories[0]
      }
    } catch {
      assets.value = {}
    }
  }

  async function fetchStrategies() {
    try {
      const { data } = await getStrategies()
      strategies.value = data.strategies || []
      if (!selectedStrategy.value && strategies.value.length > 0) {
        const pending = gridLoadedExternally.value
        await selectStrategy(strategies.value[0])
        if (pending) gridLoadedExternally.value = true
      }
    } catch {
      strategies.value = []
    }
  }

  async function selectStrategy(strategy) {
    const keepGrid = gridLoadedExternally.value
    selectedStrategy.value = strategy
    await fetchGrids()
    if (keepGrid) return
    useCustomGrid.value = false
    gridMode.value = 'rapido'
    _initCustomGridFromSchema()
  }

  function _initCustomGridFromSchema() {
    const schema = selectedStrategy.value?.schema || []
    const grid = {}
    for (const section of schema) {
      for (const field of (section.fields || [])) {
        if (field.type === 'select' && field.options) {
          grid[field.key] = [field.default]
        } else if (field.type === 'checkbox') {
          grid[field.key] = [field.default]
        } else if (field.type === 'number') {
          grid[field.key] = [field.default]
        }
      }
    }
    customGrid.value = grid
  }

  async function fetchGrids() {
    try {
      const { data } = await getOptimizerGrids(strategyFile.value)
      availableGrids.value = data.grids || {}
    } catch {
      availableGrids.value = {}
    }
  }

  async function updateComboCount() {
    const requestId = ++_countRequestId
    comboCountLoading.value = true
    comboCountError.value = null
    try {
      const btStore = useBacktestStore()
      const { data } = await getOptimizerCount(
        activeGrid.value,
        capital.value,
        strategyFile.value,
        btStore.params,
      )
      if (requestId !== _countRequestId) return
      comboCount.value = Number(data.count ?? 0)
      comboCountExact.value = data.exact !== false
      comboCountTooMany.value = Boolean(data.too_many)
      comboCountLimit.value = Number(data.max_combinations) || null
      comboCountError.value = data.message || null
    } catch (e) {
      if (requestId !== _countRequestId) return
      comboCount.value = null
      comboCountExact.value = true
      comboCountTooMany.value = false
      comboCountLimit.value = null
      comboCountError.value = e.response?.data?.error || e.message || 'Falha ao contar combinacoes'
    } finally {
      if (requestId === _countRequestId) comboCountLoading.value = false
    }
  }

  function _stopProgressPolling() {
    if (_progressTimer) {
      clearInterval(_progressTimer)
      _progressTimer = null
    }
  }

  async function run() {
    isRunning.value = true
    error.value = null
    results.value = null
    progress.value = { current: 0, total: 0, valid: 0, status: 'idle' }

    try {
      // Dispara o job — retorna imediatamente com {status: "started"}
      let startResp
      const btStore = useBacktestStore()
      if (dataSource.value === 'csv' && csvFile.value) {
        startResp = await startOptimizerCsv(
          csvFile.value,
          activeGrid.value,
          capital.value,
          minTrades.value,
          rankBy.value,
          topN.value,
          strategyFile.value,
          cycleLongMonths.value,
          cycleShortMonths.value,
          btStore.params,
        )
      } else {
        startResp = await startOptimizer({
          strategy_file: strategyFile.value,
          data_source: 'asset',
          symbol: selectedSymbol.value,
          symbol_label: selectedAssetLabel.value,
          interval: interval.value,
          grid: activeGrid.value,
          capital: capital.value,
          min_trades: minTrades.value,
          rank_by: rankBy.value,
          top_n: topN.value,
          start_date: startDate.value || null,
          end_date: endDate.value || null,
          cycle_long_months: cycleLongMonths.value,
          cycle_short_months: cycleShortMonths.value,
          config: btStore.params,
        })
      }

      if (startResp.data.error) {
        error.value = startResp.data.error
        return
      }

      // Aguarda conclusao via polling de progresso
      await new Promise((resolve) => {
        _stopProgressPolling()
        let _consecutiveErrors = 0
        let _lastCurrent = -1
        let _staleSince = Date.now()
        const _MAX_STALE_MS = 600_000  // 10min sem progresso = considera travado

        _progressTimer = setInterval(async () => {
          try {
            const { data } = await getOptimizerProgress()
            progress.value = data
            _consecutiveErrors = 0

            if (data.status === 'done' || data.status === 'error') {
              _stopProgressPolling()
              resolve()
              return
            }

            // Detecta progresso travado
            if (data.current !== _lastCurrent) {
              _lastCurrent = data.current
              _staleSince = Date.now()
            } else if (Date.now() - _staleSince > _MAX_STALE_MS) {
              _stopProgressPolling()
              progress.value = { ...data, status: 'error' }
              error.value = 'Otimizacao travada (sem progresso por 10min). Reinicie o servidor.'
              resolve()
            }
          } catch {
            _consecutiveErrors++
            if (_consecutiveErrors >= 10) {
              _stopProgressPolling()
              error.value = 'Servidor inacessivel. Verifique se o backend esta rodando.'
              resolve()
            }
          }
        }, 500)
      })

      // Busca resultado
      if (progress.value.status === 'done') {
        const resultResp = await getOptimizerResult()
        results.value = resultResp.data
      } else {
        try {
          const errResp = await getOptimizerResult()
          error.value = errResp.data?.error || 'Otimizacao falhou no servidor'
        } catch (fetchErr) {
          error.value = fetchErr.response?.data?.error || 'Otimizacao falhou no servidor'
        }
      }
    } catch (e) {
      error.value = e.response?.data?.error || e.message
    } finally {
      _stopProgressPolling()
      isRunning.value = false
    }
  }

  async function stop() {
    try {
      await stopOptimizer()
    } catch { /* ignore */ }
    // Para o polling imediatamente — nao espera o backend confirmar
    _stopProgressPolling()
    progress.value = { ...progress.value, status: 'done' }
    // Busca resultados parciais (o backend salva o que foi computado ate o cancel)
    try {
      const resultResp = await getOptimizerResult()
      if (resultResp.data && !resultResp.data.error) {
        results.value = resultResp.data
      }
    } catch { /* ignore */ }
    isRunning.value = false
  }

  function sendBestToBacktest() {
    if (!results.value?.best) return false
    const best = results.value.best
    const schema = selectedStrategy.value?.schema || []

    // Mapa reverso: label -> key
    const reverseMap = {}
    for (const section of schema) {
      for (const field of (section.fields || [])) {
        const label = field.label || field.key
        reverseMap[label] = field.key
      }
    }

    // Extrai params com chaves raw
    const paramLabels = results.value.param_columns || []
    const rawParams = {}
    for (const label of paramLabels) {
      const key = reverseMap[label] || label
      if (best[label] !== undefined) {
        rawParams[key] = best[label]
      }
    }

    const btStore = useBacktestStore()
    btStore.pendingParams = {
      ...rawParams,
      strategy_file: strategyFile.value,
      _symbol: selectedSymbol.value,
      _symbolLabel: selectedAssetLabel.value,
      _interval: interval.value,
      _dataSource: dataSource.value,
      _capital: capital.value,
      autoRun: false,
    }
    return true
  }

  function loadBestToGrid() {
    if (!results.value?.best) return false
    const best = results.value.best
    const schema = selectedStrategy.value?.schema || []

    // Mapa reverso: label -> key
    const reverseMap = {}
    const typeMap = {}
    for (const section of schema) {
      for (const field of (section.fields || [])) {
        const label = field.label || field.key
        reverseMap[label] = field.key
        typeMap[field.key] = field.type
      }
    }

    const paramLabels = results.value.param_columns || []
    const newGrid = { ...customGrid.value }
    for (const label of paramLabels) {
      const key = reverseMap[label] || label
      const val = best[label]
      if (val === undefined) continue
      if (typeMap[key] === 'checkbox') {
        newGrid[key] = [val === 'SIM' || val === true]
      } else if (typeMap[key] === 'number') {
        newGrid[key] = [Number(val)]
      } else {
        newGrid[key] = [val]
      }
    }
    customGrid.value = newGrid
    useCustomGrid.value = true
    return true
  }

  async function loadRangesFromDashboard(summaryTable) {
    if (!summaryTable || summaryTable.length === 0) return false

    // Garante que estrategias estao carregadas
    if (strategies.value.length === 0) {
      await fetchStrategies()
    }
    if (!selectedStrategy.value && strategies.value.length > 0) {
      selectedStrategy.value = strategies.value[0]
      await fetchGrids()
    }

    const schema = selectedStrategy.value?.schema || []
    if (schema.length === 0) return false

    // Mapa de tipo por key do schema
    const typeMap = {}
    for (const section of schema) {
      for (const field of (section.fields || [])) {
        typeMap[field.key] = field.type
      }
    }

    const newGrid = {}
    // Inicializa com defaults do schema
    for (const section of schema) {
      for (const field of (section.fields || [])) {
        if (field.type === 'select' && field.options) {
          newGrid[field.key] = [field.default]
        } else if (field.type === 'checkbox') {
          newGrid[field.key] = [field.default]
        } else if (field.type === 'number') {
          newGrid[field.key] = [field.default]
        }
      }
    }

    // Sobrescreve com as faixas recomendadas
    for (const row of summaryTable) {
      const key = row.optimizer_key
      if (!key || !(key in typeMap)) continue

      if (row.type === 'categorical') {
        const schemaType = typeMap[key]
        if (schemaType === 'checkbox') {
          newGrid[key] = row.values.map(v =>
            v === 'True' || v === 'true' || v === true || v === 'SIM'
          )
        } else {
          newGrid[key] = [...row.values]
        }
      } else if (row.type === 'numeric') {
        const fieldStep = _getFieldStep(schema, key)
        const arr = []
        const step = Math.max(fieldStep, 0.001)
        for (let v = row.min; v <= row.max + step * 0.01; v += step) {
          arr.push(Math.round(v * 1000) / 1000)
        }
        if (arr.length === 0) arr.push(row.min)
        newGrid[key] = arr
      }
    }
    customGrid.value = newGrid
    useCustomGrid.value = true
    gridLoadedExternally.value = true
    return true
  }

  function _getFieldStep(schema, key) {
    for (const section of schema) {
      for (const field of (section.fields || [])) {
        if (field.key === key) return field.step || 1
      }
    }
    return 1
  }

  function downloadCsv() {
    if (!results.value?.csv_data) return
    const blob = new Blob([results.value.csv_data], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `otimizacao_${results.value.symbol || 'resultado'}_${results.value.interval || ''}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return {
    assets, strategies, selectedStrategy, availableGrids,
    dataSource, selectedCategory, selectedAssetLabel, selectedSymbol,
    interval, csvFile, startDate, endDate,
    gridMode, customGrid, useCustomGrid, activeGrid, strategyFile, gridLoadedExternally,
    cycleLongMonths, cycleShortMonths,
    capital, minTrades, rankBy, topN,
    isRunning, comboCount, comboCountLoading, comboCountError,
    comboCountExact, comboCountTooMany, comboCountLimit,
    results, error, progress,
    fetchAssets, fetchStrategies, selectStrategy, fetchGrids,
    updateComboCount, run, stop, sendBestToBacktest, loadBestToGrid,
    loadRangesFromDashboard, downloadCsv,
  }
})
