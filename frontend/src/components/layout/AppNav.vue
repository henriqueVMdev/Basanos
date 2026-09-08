<template>
  <nav aria-label="Navegação principal"
       class="h-14 bg-surface-800/90 backdrop-blur border-b border-surface-500 flex items-center px-4 gap-4 shrink-0 sticky top-0 z-40">

    <!-- Marca -->
    <RouterLink to="/" class="flex items-center gap-2.5 pr-2 group">
      <img
        src="/basanos-logo.png"
        alt="Basanos"
        class="w-8 h-8 object-contain rounded-md shadow-yellow-glow-sm
               group-hover:shadow-yellow-glow transition-shadow"
      />
      <div class="leading-none hidden lg:block">
        <div class="text-[13px] font-bold tracking-wide text-gray-100">BASANOS</div>
        <div class="text-[11px] text-gray-500 uppercase tracking-[0.2em]">quant lab</div>
      </div>
    </RouterLink>

    <div class="w-px h-6 bg-surface-500 hidden md:block" />

    <!-- Nav agrupada: dropdown por área -->
    <div ref="navEl" class="flex items-center gap-1">
      <div v-for="g in GROUPS" :key="g.label" class="relative">
        <button
          :ref="(el) => { if (el) groupBtns[g.label] = el }"
          type="button"
          :aria-expanded="openGroup === g.label"
          aria-haspopup="true"
          :aria-controls="`nav-grupo-${slug(g.label)}`"
          @click="openGroup = openGroup === g.label ? null : g.label"
          class="nav-link"
          :class="{ active: groupActive(g) || openGroup === g.label }"
        >
          <svg viewBox="0 0 24 24" class="w-3.5 h-3.5 shrink-0" fill="none"
               stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round" v-html="g.icon" />
          <span class="hidden md:inline">{{ g.label }}</span>
          <svg viewBox="0 0 24 24" class="w-2.5 h-2.5 shrink-0 transition-transform"
               :class="{ 'rotate-180': openGroup === g.label }" fill="none"
               stroke="currentColor" stroke-width="2.5" stroke-linecap="round"
               stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
        </button>

        <Transition name="dd">
          <div v-if="openGroup === g.label"
               :id="`nav-grupo-${slug(g.label)}`"
               class="absolute left-0 top-full mt-1.5 min-w-48 rounded-lg border
                      border-surface-500 bg-surface-800/95 backdrop-blur shadow-xl
                      py-1.5 z-50">
            <div class="px-3 pb-1 text-[11px] uppercase tracking-[0.18em] text-gray-500">
              {{ g.hint }}
            </div>
            <RouterLink
              v-for="item in g.items" :key="item.to" :to="item.to"
              @click="openGroup = null"
              class="flex items-center gap-2.5 px-3 py-1.5 text-[12px] transition-colors"
              :class="route.path === item.to
                ? 'text-accent-yellow bg-accent-yellow/10'
                : 'text-gray-400 hover:text-gray-100 hover:bg-surface-600'"
            >
              <svg viewBox="0 0 24 24" class="w-3.5 h-3.5 shrink-0" fill="none"
                   stroke="currentColor" stroke-width="2" stroke-linecap="round"
                   stroke-linejoin="round" v-html="item.icon" />
              {{ item.label }}
            </RouterLink>
          </div>
        </Transition>
      </div>
    </div>

    <div class="flex-1" />

    <!-- Status de execução. O rotulo some abaixo de md, mas a regiao viva
         continua anunciando: e a unica pista de que algo esta rodando. -->
    <div class="flex items-center gap-2 text-xs text-accent-yellow/80 shrink-0"
         role="status" aria-live="polite">
      <template v-if="busyLabel">
        <span class="dollar-loader-sm" aria-hidden="true">$</span>
        <span class="hidden md:inline">{{ busyLabel }}</span>
        <span class="md:hidden sr-only">{{ busyLabel }}</span>
      </template>
    </div>

    <!-- Command line hint -->
    <button @click="$emit('open-cmd')" title="Command line (Ctrl+K)"
            class="hidden md:flex items-center gap-1.5 text-[11px] font-mono text-gray-500
                   border border-surface-500 rounded-md px-2 py-1 hover:text-accent-yellow
                   hover:border-accent-yellow/40 transition-colors shrink-0" aria-label="Command line (Ctrl+K)">
      <span class="text-accent-yellow">&gt;_</span> Ctrl+K
    </button>

    <!-- Sino de alertas -->
    <RouterLink to="/alerts" class="hit-target shrink-0 text-gray-400 hover:text-accent-yellow transition-colors"
                :aria-label="alertsLabel">
      <svg viewBox="0 0 24 24" class="w-4.5 h-4.5 w-[18px] h-[18px]" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
        <path d="M13.73 21a2 2 0 0 1-3.46 0"/>
      </svg>
      <span v-if="terminal.unseenTriggered.length" aria-hidden="true"
            class="absolute -top-1.5 -right-1.5 min-w-[14px] h-[14px] px-0.5 rounded-full
                   bg-accent-red text-white text-[11px] font-bold flex items-center justify-center">
        {{ terminal.unseenTriggered.length }}
      </span>
    </RouterLink>

    <!-- Relógio de sessão (UTC + Brasília) -->
    <div class="hidden md:flex items-center gap-3 font-mono text-[11px] leading-none shrink-0">
      <div class="text-right">
        <div class="text-gray-300">{{ clockUtc }} <span class="text-gray-500">UTC</span></div>
        <div class="text-gray-400">{{ clockBrt }} <span class="text-gray-500">BRT</span></div>
      </div>
      <!-- Janela de entrada 19h-00h BRT da estratégia validada -->
      <div class="flex items-center gap-1.5 px-2 py-1 rounded-md border"
           :class="inWindow
             ? 'border-accent-yellow/40 bg-accent-yellow/10 text-accent-yellow'
             : 'border-surface-500 bg-surface-700 text-gray-500'"
           title="Janela de entrada da estratégia validada: 19:00-00:00 Brasília">
        <span class="w-1.5 h-1.5 rounded-full"
              :class="inWindow ? 'bg-accent-yellow animate-pulse' : 'bg-gray-600'" />
        <span class="text-[11px] font-semibold tracking-wide">19-00h</span>
        <span class="sr-only">Janela de entrada da estratégia validada: 19:00 às 00:00 de Brasília</span>
      </div>
    </div>
  </nav>
</template>

<script setup>
import { quandoVisivel } from '@/composables/visibilidade.js'
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import { useDashboardStore } from '@/stores/dashboard.js'
import { useBacktestStore } from '@/stores/backtest.js'
import { useOptimizerStore } from '@/stores/optimizer.js'
import { usePropChallengeStore } from '@/stores/propChallenge.js'
import { useRegimeStore } from '@/stores/regime.js'
import { useTerminalStore } from '@/stores/terminal.js'

defineEmits(['open-cmd'])

const route = useRoute()
const terminal = useTerminalStore()
const dashStore = useDashboardStore()
const btStore = useBacktestStore()
const optStore = useOptimizerStore()
const propStore = usePropChallengeStore()
const regimeStore = useRegimeStore()

// Ícones estilo feather (conteúdo interno do <svg>)
const I = {
  params: '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
  backtest: '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
  chart: '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
  optimizer: '<circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/>',
  prop: '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
  regime: '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
  journal: '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
  auto: '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
  monitor: '<circle cx="12" cy="12" r="3"/><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/>',
  screener: '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
  eqs: '<line x1="19" y1="5" x2="5" y2="19"/><circle cx="6.5" cy="6.5" r="2.5"/><circle cx="17.5" cy="17.5" r="2.5"/>',
  company: '<path d="M3 21h18"/><path d="M5 21V7l8-4v18"/><path d="M19 21V11l-6-4"/><line x1="9" y1="9" x2="9" y2="9.01"/><line x1="9" y1="12" x2="9" y2="12.01"/><line x1="9" y1="15" x2="9" y2="15.01"/>',
  rates: '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
  tech: '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/><circle cx="19" cy="5" r="2"/>',
  trade: '<circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/>',
  alt: '<circle cx="12" cy="12" r="2"/><path d="M16.24 7.76a6 6 0 0 1 0 8.49m-8.48-.01a6 6 0 0 1 0-8.49m11.31-2.82a10 10 0 0 1 0 14.14m-14.14 0a10 10 0 0 1 0-14.14"/>',
  cdty: '<path d="M12 2a10 10 0 1 0 10 10"/><path d="M12 2a10 10 0 0 1 10 10"/><path d="M2 12h20"/><path d="M12 2c2.5 2.7 4 6.2 4 10s-1.5 7.3-4 10c-2.5-2.7-4-6.2-4-10s1.5-7.3 4-10z"/>',
  news: '<path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2"/><line x1="18" y1="14" x2="10" y2="14"/><line x1="18" y1="18" x2="10" y2="18"/><line x1="10" y1="6" x2="18" y2="6"/><line x1="10" y1="10" x2="18" y2="10"/>',
  seasonal: '<circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/><path d="M5.6 5.6l12.8 12.8M18.4 5.6L5.6 18.4"/>',
}

// Grupos por afinidade: pesquisa de estratégia → dados de mercado →
// análise fundamental/macro → execução e acompanhamento
const GROUPS = [
  { label: 'Estratégia', hint: 'pesquisa & validação', icon: I.backtest,
    items: [
      { to: '/dashboard', label: 'Parâmetros', icon: I.params },
      { to: '/backtest', label: 'Backtesting', icon: I.backtest },
      { to: '/optimizer', label: 'Otimizador', icon: I.optimizer },
      { to: '/regime', label: 'Regimes', icon: I.regime },
      { to: '/prop-challenge', label: 'Prop Challenge', icon: I.prop },
    ] },
  { label: 'Mercados', hint: 'tempo real & gráficos', icon: I.monitor,
    items: [
      { to: '/monitor', label: 'Monitor', icon: I.monitor },
      { to: '/screener', label: 'Screener', icon: I.screener },
      { to: '/grafico', label: 'Gráfico', icon: I.chart },
      { to: '/tech', label: 'Análise Técnica', icon: I.tech },
      { to: '/news', label: 'Notícias', icon: I.news },
      { to: '/seasonality', label: 'Sazonalidade', icon: I.seasonal },
      { to: '/intelligence', label: 'Central de Sinais', icon: I.regime },
      { to: '/calendar', label: 'Calendário de Eventos', icon: I.journal },
      { to: '/portfolio-lab', label: 'Portfolio Lab', icon: I.optimizer },
    ] },
  { label: 'Análise', hint: 'fundamental & macro', icon: I.company,
    items: [
      { to: '/eqs', label: 'EQS · Screening', icon: I.eqs },
      { to: '/ea', label: 'Empresa (FA)', icon: I.company },
      { to: '/rates', label: 'Juros & Crédito', icon: I.rates },
      { to: '/cdty', label: 'Commodities', icon: I.cdty },
      { to: '/alt', label: 'Alt Data & On-chain', icon: I.alt },
      { to: '/degenerado', label: 'Degenerado', icon: I.alt },
    ] },
  { label: 'Execução', hint: 'ordens & acompanhamento', icon: I.trade,
    items: [
      { to: '/trade', label: 'Trading (OMS)', icon: I.trade },
      { to: '/automation', label: 'Automação', icon: I.auto },
      { to: '/agentes', label: 'Agentes IA', icon: I.auto },
      { to: '/hft', label: 'HFT On-chain', icon: I.auto },
      { to: '/journal', label: 'Diário', icon: I.journal },
    ] },
]

const openGroup = ref(null)
const navEl = ref(null)
const groupBtns = {}

const slug = (s) => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '')

const alertsLabel = computed(() => {
  const n = terminal.unseenTriggered.length
  return n ? `Alertas, ${n} ${n === 1 ? 'novo' : 'novos'}` : 'Alertas'
})

function groupActive(g) {
  return g.items.some((i) => i.to === route.path)
}

function onClickOutside(e) {
  if (openGroup.value && navEl.value && !navEl.value.contains(e.target)) {
    openGroup.value = null
  }
}

// Esc fecha e devolve o foco ao botao que abriu, senao o foco fica orfao no
// meio do documento depois que o painel some.
function onKeydown(e) {
  if (e.key !== 'Escape' || !openGroup.value) return
  const btn = groupBtns[openGroup.value]
  openGroup.value = null
  btn?.focus()
}

const busyLabel = computed(() => {
  if (regimeStore.isRunning) return 'Detectando regimes...'
  if (propStore.isRunning) return 'Simulando prop...'
  if (optStore.isRunning) return 'Otimizando...'
  if (btStore.isRunning) return 'Executando backtest...'
  if (dashStore.loading) return 'Carregando...'
  return ''
})

// ── Relógio de sessão ──────────────────────────────────────────────
const clockUtc = ref('')
const clockBrt = ref('')
const inWindow = ref(false)
let clockTimer = null

function tick() {
  const now = new Date()
  const pad = (v) => String(v).padStart(2, '0')
  clockUtc.value = `${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())}`
  const brtH = (now.getUTCHours() + 21) % 24 // UTC-3
  clockBrt.value = `${pad(brtH)}:${pad(now.getUTCMinutes())}`
  // Janela de entrada validada: 19:00-00:00 BRT = 22:00-03:00 UTC
  const h = now.getUTCHours()
  inWindow.value = h >= 22 || h < 3
}

onMounted(() => {
  tick()
  clockTimer = setInterval(quandoVisivel(tick), 1000)
  terminal.startAlertsPolling()      // badge do sino em qualquer página
  window.addEventListener('click', onClickOutside)
  window.addEventListener('keydown', onKeydown)
})
onBeforeUnmount(() => {
  clearInterval(clockTimer)
  window.removeEventListener('click', onClickOutside)
  window.removeEventListener('keydown', onKeydown)
})
</script>

<style scoped>
.nav-link {
  @apply flex items-center gap-1.5 px-2.5 py-1.5 text-[13px] text-gray-400 rounded-lg
         hover:text-gray-200 hover:bg-surface-600 transition-all duration-150 font-medium
         whitespace-nowrap;
}
.nav-link.active {
  @apply text-accent-yellow bg-accent-yellow/10;
}

/* dropdown: fade + leve descida */
.dd-enter-active,
.dd-leave-active {
  transition: opacity 0.12s ease, transform 0.12s ease;
}
.dd-enter-from,
.dd-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
</style>
