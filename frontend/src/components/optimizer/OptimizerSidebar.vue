<template>
  <aside class="h-full bg-surface-800 border-r border-surface-500 overflow-y-auto p-4 space-y-4 text-sm">
    <!-- Estrategia -->
    <section>
      <h3 class="sidebar-section-title">Estrategia</h3>
      <div v-if="store.strategies.length === 0" class="text-xs text-gray-400 py-2">
        Nenhuma estrategia encontrada em strategies/
      </div>
      <template v-else>
        <select
          :value="store.selectedStrategy?.file"
          @change="onStrategyChange($event.target.value)"
          aria-label="Estrategia"
          class="form-select w-full mb-2"
        >
          <option
            v-for="s in store.strategies"
            :key="s.file"
            :value="s.file"
            :disabled="!!s.error"
          >{{ s.name }}</option>
        </select>
        <p v-if="store.selectedStrategy?.description" class="text-xs text-gray-400 leading-relaxed mb-2">
          {{ store.selectedStrategy.description }}
        </p>
      </template>
    </section>

    <!-- Fonte de dados -->
    <section>
      <h3 class="sidebar-section-title">Dados</h3>
      <div class="flex gap-2 mb-3">
        <button
          class="flex-1 py-1.5 text-xs rounded-lg font-medium transition-colors"
          :class="store.dataSource === 'asset' ? 'bg-accent-yellow text-black font-bold' : 'bg-surface-500 text-gray-400'"
          @click="store.dataSource = 'asset'"
        >Ativo</button>
        <button
          class="flex-1 py-1.5 text-xs rounded-lg font-medium transition-colors"
          :class="store.dataSource === 'csv' ? 'bg-accent-yellow text-black font-bold' : 'bg-surface-500 text-gray-400'"
          @click="store.dataSource = 'csv'"
        >CSV</button>
      </div>

      <template v-if="store.dataSource === 'asset'">
        <label class="text-xs text-gray-400 block mb-1" for="f-components-optimizer-optimizersidebar-vue-1">Categoria</label>
        <select id="f-components-optimizer-optimizersidebar-vue-1" v-model="store.selectedCategory" class="form-select w-full mb-2">
          <option v-for="cat in Object.keys(store.assets)" :key="cat" :value="cat">{{ cat }}</option>
        </select>

        <label class="text-xs text-gray-400 block mb-1" for="f-components-optimizer-optimizersidebar-vue-2">Ativo</label>
        <select id="f-components-optimizer-optimizersidebar-vue-2" v-model="selectedAssetKey" class="form-select w-full mb-2">
          <option value="">Selecione...</option>
          <option v-for="(ticker, name) in categoryAssets" :key="ticker" :value="name">{{ name }}</option>
        </select>

        <label class="text-xs text-gray-400 block mb-1" for="f-components-optimizer-optimizersidebar-vue-3">Timeframe</label>
        <select id="f-components-optimizer-optimizersidebar-vue-3" v-model="store.interval" class="form-select w-full mb-2">
          <option v-for="tf in ['15m', '30m', '1h', '2h', '4h', '1d', '1wk', '1mo']" :key="tf" :value="tf">{{ tf }}</option>
        </select>

        <div class="grid grid-cols-2 gap-2">
          <div>
            <label class="text-xs text-gray-400 block mb-1" for="f-components-optimizer-optimizersidebar-vue-4">Inicio</label>
            <input id="f-components-optimizer-optimizersidebar-vue-4" type="date" v-model="store.startDate" class="form-input w-full" />
          </div>
          <div>
            <label class="text-xs text-gray-400 block mb-1" for="f-components-optimizer-optimizersidebar-vue-5">Fim</label>
            <input id="f-components-optimizer-optimizersidebar-vue-5" type="date" v-model="store.endDate" class="form-input w-full" />
          </div>
        </div>
      </template>

      <template v-else>
        <label class="text-xs text-gray-400 block mb-1" for="f-components-optimizer-optimizersidebar-vue-6">Upload CSV</label>
        <input id="f-components-optimizer-optimizersidebar-vue-6" type="file" accept=".csv" @change="onFileChange" class="form-input w-full text-xs" />
      </template>
    </section>

    <!-- Grid -->
    <section>
      <h3 class="sidebar-section-title">Grid de Parametros</h3>
      <label class="text-xs text-gray-400 block mb-1" for="f-components-optimizer-optimizersidebar-vue-7">Modo</label>
      <select id="f-components-optimizer-optimizersidebar-vue-7" v-model="store.gridMode" class="form-select w-full mb-2" @change="store.useCustomGrid = false">
        <option v-for="mode in gridModes" :key="mode" :value="mode">{{ mode }}</option>
      </select>

      <label class="flex items-center gap-2 text-gray-400 mb-2 cursor-pointer">
        <input type="checkbox" v-model="store.useCustomGrid" class="accent-accent-yellow" />
        Personalizar grid
      </label>

      <template v-if="store.useCustomGrid && schemaFields.length > 0">
        <div v-for="field in schemaFields" :key="field.key" class="mb-2">

          <!-- select / checkbox: dropdown multi-select -->
          <template v-if="field.type === 'select' || field.type === 'checkbox'">
            <label class="text-xs text-gray-400 block mb-1">{{ field.label }}</label>
            <div class="relative" v-click-outside="() => closeDropdown(field.key)">
              <button
                @click="toggleDropdown(field.key)"
                class="form-input w-full text-left flex items-center justify-between cursor-pointer"
              >
                <span class="truncate text-xs">{{ dropdownLabel(field) }}</span>
                <svg class="w-3 h-3 shrink-0 text-gray-400 transition-transform" :class="openDropdown === field.key ? 'rotate-180' : ''" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                </svg>
              </button>
              <div
                v-if="openDropdown === field.key"
                class="absolute z-30 left-0 right-0 mt-1 bg-surface-700 border border-surface-500 rounded-md shadow-lg max-h-48 overflow-y-auto"
              >
                <label
                  v-for="opt in dropdownOptions(field)"
                  :key="String(opt.value)"
                  class="flex items-center gap-2 px-3 py-1.5 hover:bg-surface-600 cursor-pointer text-xs"
                >
                  <input
                    type="checkbox"
                    :checked="isSelected(field.key, opt.value)"
                    @change="toggleArrayItem(field.key, opt.value)"
                    class="accent-accent-yellow"
                  />
                  <span class="text-gray-300">{{ opt.label }}</span>
                </label>
              </div>
            </div>
          </template>

          <!-- number: min/max/step range -->
          <template v-else-if="field.type === 'number'">
            <label class="text-xs text-gray-400 block mb-1">{{ field.label }}</label>
            <div class="grid grid-cols-3 gap-1">
              <input type="number" v-model.number="ranges[field.key].min" class="form-input w-full text-xs" placeholder="min" />
              <input type="number" v-model.number="ranges[field.key].max" class="form-input w-full text-xs" placeholder="max" />
              <input type="number" v-model.number="ranges[field.key].step" class="form-input w-full text-xs" placeholder="step" />
            </div>
            <div class="text-xs text-gray-500 mt-0.5">
              {{ (store.customGrid[field.key] || []).length }} valores
            </div>
          </template>

        </div>
      </template>

      <div v-else-if="store.useCustomGrid && schemaFields.length === 0" class="text-xs text-gray-400">
        Estrategia sem schema de parametros
      </div>
    </section>

    <!-- Backtest config -->
    <section>
      <h3 class="sidebar-section-title">Configuracao</h3>
      <label class="text-xs text-gray-400 block mb-1" for="f-components-optimizer-optimizersidebar-vue-8">Capital Inicial</label>
      <input id="f-components-optimizer-optimizersidebar-vue-8" type="number" v-model.number="store.capital" class="form-input w-full mb-2" />

      <label class="text-xs text-gray-400 block mb-1" for="f-components-optimizer-optimizersidebar-vue-9">Min Trades</label>
      <input id="f-components-optimizer-optimizersidebar-vue-9" type="number" v-model.number="store.minTrades" class="form-input w-full mb-2" />

      <label class="text-xs text-gray-400 block mb-1" for="f-components-optimizer-optimizersidebar-vue-10">Rankear por</label>
      <select id="f-components-optimizer-optimizersidebar-vue-10" v-model="store.rankBy" class="form-select w-full mb-2">
        <option v-for="r in ['Score','Retorno (%)','Sharpe','Sortino','Calmar','Omega','Sterling','Burke','Profit Factor','Win Rate (%)']" :key="r" :value="r">{{ r }}</option>
      </select>

      <label class="text-xs text-gray-400 block mb-1" for="f-components-optimizer-optimizersidebar-vue-11">Top-N</label>
      <input id="f-components-optimizer-optimizersidebar-vue-11" type="number" v-model.number="store.topN" min="5" max="200" class="form-input w-full mb-2" />
    </section>

    <!-- Ciclo Sazonal -->
    <section>
      <button
        @click="showCycleMonths = !showCycleMonths"
        class="w-full flex items-center justify-between py-1.5 text-xs font-bold text-gray-400 uppercase tracking-wider hover:text-gray-300 transition-colors cursor-pointer"
      >
        <span>Ciclo Sazonal</span>
        <svg class="w-3.5 h-3.5 transition-transform" :class="showCycleMonths ? 'rotate-180' : ''" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
        </svg>
      </button>
      <div v-if="showCycleMonths" class="space-y-2 mt-2">
        <div>
          <p class="text-[11px] text-accent-yellow font-semibold mb-1">LONG</p>
          <div class="grid grid-cols-6 gap-1">
            <button
              v-for="m in months"
              :key="'L' + m.n"
              @click="toggleMonth('long', m.n)"
              class="py-1 text-[11px] rounded font-medium transition-colors"
              :class="store.cycleLongMonths.includes(m.n)
                ? 'bg-accent-yellow/30 text-accent-yellow border border-accent-yellow/50'
                : 'bg-surface-600 text-gray-400 border border-surface-500'"
            >{{ m.label }}</button>
          </div>
        </div>
        <div>
          <p class="text-[11px] text-accent-red-light font-semibold mb-1">SHORT</p>
          <div class="grid grid-cols-6 gap-1">
            <button
              v-for="m in months"
              :key="'S' + m.n"
              @click="toggleMonth('short', m.n)"
              class="py-1 text-[11px] rounded font-medium transition-colors"
              :class="store.cycleShortMonths.includes(m.n)
                ? 'bg-accent-red/30 text-accent-red-light border border-accent-red/50'
                : 'bg-surface-600 text-gray-400 border border-surface-500'"
            >{{ m.label }}</button>
          </div>
        </div>
      </div>
    </section>

    <!-- Horário de Operação -->
    <HourFilter :params="btStore.params" />

    <!-- Combo count + run -->
    <section>
      <div class="flex items-center justify-between text-gray-400 mb-2">
        <span>Combinacoes:</span>
        <span v-if="store.comboCountLoading" class="text-xs text-gray-400">calculando...</span>
        <span v-else-if="store.comboCount !== null" class="font-bold text-gray-200">
          {{ store.comboCountExact ? '' : 'ate ' }}{{ store.comboCount.toLocaleString('pt-BR') }}
        </span>
        <span v-else class="font-bold text-gray-500">--</span>
      </div>
      <div v-if="store.comboCountError" class="text-accent-red-light text-xs mb-2">
        {{ store.comboCountError }}
      </div>
      <div v-else-if="store.comboCount > 50000" class="text-accent-red-light text-xs mb-2">
        Muitas combinacoes! Pode demorar bastante.
      </div>
      <!-- Progress bar -->
      <div v-if="store.isRunning && store.progress.total > 0" class="mb-2">
        <div class="flex justify-between text-xs text-gray-400 mb-1">
          <span>{{ store.progress.current }} / {{ store.progress.total }}</span>
          <span>{{ Math.round(store.progress.current / store.progress.total * 100) }}%</span>
        </div>
        <div class="w-full h-1.5 bg-surface-600 rounded-full overflow-hidden">
          <div
            class="h-full bg-accent-yellow rounded-full transition-all duration-300"
            :style="{ width: (store.progress.current / store.progress.total * 100) + '%' }"
          />
        </div>
        <div class="text-xs text-gray-400 mt-1">
          {{ store.progress.valid }} validos
        </div>
      </div>

      <!-- Load best params -->
      <button
        v-if="!store.isRunning && store.results?.best"
        @click="loadBest()"
        class="w-full py-2 rounded-lg font-semibold text-xs bg-accent-yellow/10 text-accent-yellow border border-accent-yellow/30 hover:bg-accent-yellow/20 transition-all duration-200 mb-2"
      >
        Carregar Parametros Recomendados
      </button>

      <!-- Run / Stop buttons -->
      <button
        v-if="!store.isRunning"
        @click="store.run()"
        :disabled="store.comboCountLoading || store.comboCount === null || store.comboCount === 0 || store.comboCountTooMany"
        class="w-full py-2.5 rounded-lg font-semibold text-sm bg-accent-yellow text-surface-900 hover:bg-accent-yellow/90 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Rodar Otimizacao
      </button>
      <button
        v-else
        @click="store.stop()"
        class="w-full py-2.5 rounded-lg font-semibold text-sm bg-accent-red text-white hover:bg-accent-red/80 transition-all duration-200"
      >
        Parar Otimizacao
      </button>
    </section>
  </aside>
</template>

<script setup>
import { ref, computed, watch, reactive, onMounted } from 'vue'
import { useOptimizerStore } from '@/stores/optimizer.js'
import { useBacktestStore } from '@/stores/backtest.js'
import HourFilter from '@/components/HourFilter.vue'

const store = useOptimizerStore()
// O otimizador reaproveita btStore.params como config fixa (ver optimizer.js).
const btStore = useBacktestStore()

// Dropdown state
const openDropdown = ref(null)
const showCycleMonths = ref(false)

function toggleDropdown(key) {
  openDropdown.value = openDropdown.value === key ? null : key
}

function closeDropdown(key) {
  if (openDropdown.value === key) openDropdown.value = null
}

function dropdownOptions(field) {
  if (field.type === 'select') {
    return (field.options || []).map(o => ({ value: o, label: o }))
  }
  // checkbox
  return [
    { value: true, label: 'ON' },
    { value: false, label: 'OFF' },
  ]
}

function dropdownLabel(field) {
  const selected = store.customGrid[field.key] || []
  if (selected.length === 0) return 'Nenhum'
  if (field.type === 'checkbox') {
    return selected.map(v => v ? 'ON' : 'OFF').join(', ')
  }
  return selected.join(', ')
}

function isSelected(key, value) {
  const arr = store.customGrid[key] || []
  return arr.includes(value)
}

// v-click-outside directive
const vClickOutside = {
  mounted(el, binding) {
    el._clickOutside = (e) => {
      if (!el.contains(e.target)) binding.value()
    }
    document.addEventListener('mousedown', el._clickOutside)
  },
  unmounted(el) {
    document.removeEventListener('mousedown', el._clickOutside)
  },
}

// Meses para ciclo sazonal
const months = [
  { n: 1, label: 'Jan' }, { n: 2, label: 'Fev' }, { n: 3, label: 'Mar' },
  { n: 4, label: 'Abr' }, { n: 5, label: 'Mai' }, { n: 6, label: 'Jun' },
  { n: 7, label: 'Jul' }, { n: 8, label: 'Ago' }, { n: 9, label: 'Set' },
  { n: 10, label: 'Out' }, { n: 11, label: 'Nov' }, { n: 12, label: 'Dez' },
]

function toggleMonth(type, monthNum) {
  const arr = type === 'long' ? store.cycleLongMonths : store.cycleShortMonths
  const idx = arr.indexOf(monthNum)
  if (idx >= 0) arr.splice(idx, 1)
  else arr.push(monthNum)
}

// Grid modes disponiveis
const gridModes = computed(() => {
  const modes = Object.keys(store.availableGrids)
  return modes.length > 0 ? modes : ['rapido', 'completo', 'custom']
})

// Schema fields da estrategia selecionada (flat list, sem campos ocultos no optimizer)
const schemaFields = computed(() => {
  const schema = store.selectedStrategy?.schema || []
  const fields = []
  for (const section of schema) {
    for (const field of (section.fields || [])) {
      if (field.optimizer_hidden) continue
      fields.push(field)
    }
  }
  return fields
})

// Ranges para campos numericos (min/max/step)
const ranges = reactive({})
let _skipRangeSync = false

watch(schemaFields, (fields) => {
  // Se grid foi carregado externamente, inicializa ranges a partir do grid
  if (store.gridLoadedExternally) {
    _skipRangeSync = true
    for (const field of fields) {
      if (field.type !== 'number') continue
      const vals = store.customGrid[field.key]
      if (vals && vals.length > 0) {
        ranges[field.key] = {
          min: minOf(vals),
          max: maxOf(vals),
          step: field.step || 1,
        }
      } else {
        const def = field.default ?? 0
        ranges[field.key] = { min: def, max: def, step: field.step || 1 }
      }
    }
    store.gridLoadedExternally = false
    _skipRangeSync = false
    return
  }
  for (const field of fields) {
    if (field.type === 'number' && !ranges[field.key]) {
      const def = field.default ?? 0
      ranges[field.key] = {
        min: def,
        max: def,
        step: field.step || 1,
      }
    }
  }
}, { immediate: true })

// Quando ranges mudam, atualiza o customGrid com os valores gerados
watch(ranges, () => {
  if (_skipRangeSync) return
  for (const field of schemaFields.value) {
    if (field.type !== 'number') continue
    const r = ranges[field.key]
    if (!r) continue
    const min = Number(r.min)
    const max = Number(r.max)
    const requestedStep = Number(r.step)
    if (!Number.isFinite(min) || !Number.isFinite(max)
        || !Number.isFinite(requestedStep) || requestedStep <= 0 || min > max) {
      store.customGrid[field.key] = []
      continue
    }
    const arr = []
    const step = Math.max(requestedStep, 0.001)
    for (let v = min; v <= max + step * 0.01; v += step) {
      arr.push(Math.round(v * 1000) / 1000)
    }
    if (arr.length === 0) arr.push(min)
    store.customGrid[field.key] = arr
  }
}, { deep: true })

// Strategy change
function onStrategyChange(file) {
  const strat = store.strategies.find(s => s.file === file)
  if (strat) {
    // Campos com a mesma key em estrategias diferentes nao devem reaproveitar
    // ranges antigos e sobrescrever o novo grid logo apos a troca.
    for (const key of Object.keys(ranges)) delete ranges[key]
    store.selectStrategy(strat)
  }
}

// Selected asset
const selectedAssetKey = ref('')
const categoryAssets = computed(() => store.assets[store.selectedCategory] || {})

watch(selectedAssetKey, (name) => {
  const ticker = categoryAssets.value[name]
  if (ticker) {
    store.selectedAssetLabel = name
    store.selectedSymbol = ticker
  }
})

// Pré-seleciona os dropdowns com o ativo escolhido em outra página
import { useWorkspaceStore } from '@/stores/workspace.js'
import { maxOf, minOf } from '@/utils.js'
const ws = useWorkspaceStore()
watch(() => store.assets, (a) => {
  if (selectedAssetKey.value || !store.selectedSymbol) return
  const hit = ws.findAsset(a, store.selectedSymbol)
  if (hit) {
    store.selectedCategory = hit.category
    selectedAssetKey.value = hit.label
  }
}, { immediate: true })

// File change
function onFileChange(e) {
  store.csvFile = e.target.files[0] || null
}

// Toggle array item helper
function toggleArrayItem(key, item) {
  if (!store.customGrid[key]) store.customGrid[key] = []
  const arr = store.customGrid[key]
  const idx = arr.indexOf(item)
  // Todo parametro precisa de ao menos um valor. Uma lista vazia era omitida
  // pelo backend e podia herdar configuracao antiga sem o usuario perceber.
  if (idx >= 0) {
    if (arr.length > 1) arr.splice(idx, 1)
    return
  }
  arr.push(item)
}

function loadBest() {
  if (!store.loadBestToGrid()) return
  syncRangesFromGrid()
}

function syncRangesFromGrid() {
  for (const field of schemaFields.value) {
    if (field.type !== 'number') continue
    const vals = store.customGrid[field.key]
    if (!vals || vals.length === 0) continue
    const min = minOf(vals)
    const max = maxOf(vals)
    ranges[field.key] = {
      min,
      max,
      step: field.step || 1,
    }
  }
}

// Update combo count when grid changes
let countTimer = null
watch(() => store.activeGrid, () => {
  clearTimeout(countTimer)
  countTimer = setTimeout(() => store.updateComboCount(), 300)
}, { deep: true })

onMounted(() => {
  store.updateComboCount()
})
</script>
