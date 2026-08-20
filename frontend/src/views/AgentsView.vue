<template>
  <div class="flex h-[calc(100dvh-3.5rem)] overflow-hidden">

    <!-- Sidebar: lista + editor -->
    <div class="w-80 shrink-0 border-r border-surface-500 overflow-y-auto p-4 space-y-3">
      <div class="flex items-center justify-between">
        <h1 class="text-sm font-bold text-gray-100">
          <span class="text-accent-yellow">◆</span> Agentes
        </h1>
        <button @click="startNew" class="px-2 py-1 text-xs rounded-md border border-surface-600 text-gray-400 hover:text-accent-yellow transition-colors">
          + Novo
        </button>
      </div>

      <div v-if="!store.hasKey" class="card p-2.5 text-xs text-accent-yellow/80 border-accent-yellow/30">
        Sem <code>OPENROUTER_API_KEY</code> no .env — runtime nativo não vai responder. Configure ou use base_url local (Ollama).
      </div>

      <!-- Lista -->
      <div
        v-for="a in store.agents" :key="a.id"
        class="card p-3 cursor-pointer transition-colors"
        :class="store.selected?.id === a.id ? 'border-accent-yellow/50' : 'hover:bg-surface-700/40'"
        @click="store.select(a)"
      >
        <div class="flex items-center justify-between">
          <span class="text-sm font-semibold text-gray-100">{{ a.name }}</span>
          <span class="text-[11px] uppercase tracking-wider px-1.5 py-0.5 rounded"
                :class="a.agent_type === 'native' ? 'bg-accent-yellow/10 text-accent-yellow' : 'bg-surface-600 text-gray-400'">
            {{ a.agent_type }}
          </span>
        </div>
        <div class="text-[11px] text-gray-500 mt-0.5 font-mono">{{ a.model }}</div>
        <div class="flex gap-2 mt-1.5">
          <button @click.stop="editAgent(a)" class="text-[11px] text-gray-400 hover:text-gray-200">editar</button>
          <button @click.stop="store.removeAgent(a.id)" class="text-[11px] text-gray-400 hover:text-accent-red-light">excluir</button>
        </div>
      </div>

      <!-- Editor -->
      <div v-if="editing" class="card p-3 space-y-2.5">
        <input v-model="form.name" placeholder="Nome" class="inp" />
        <textarea v-model="form.system_prompt" placeholder="System prompt" rows="5" class="inp resize-y" />
        <input v-model="form.model" placeholder="Modelo (ex: anthropic/claude-sonnet-5)" class="inp font-mono text-[11px]" />
        <div class="flex gap-2">
          <select v-model="form.agent_type" class="inp flex-1">
            <option value="native">native (loop basanos)</option>
            <option value="hermes">hermes (gateway)</option>
            <option value="openclaw">openclaw (gateway)</option>
          </select>
          <input v-model.number="form.temperature" type="number" step="0.1" min="0" max="2" class="inp w-16" title="Temperatura"  aria-label="Temperatura"/>
        </div>
        <input v-model="form.base_url" placeholder="Base URL (vazio = OpenRouter)" class="inp font-mono text-[11px]" />
        <div v-if="form.agent_type === 'native'">
          <div class="text-[11px] text-gray-400 mb-1">Tools:</div>
          <label v-for="(desc, name) in store.toolCatalog" :key="name" class="flex items-start gap-1.5 text-[11px] text-gray-400 mb-1 cursor-pointer" :title="desc">
            <input type="checkbox" :value="name" v-model="form.enabled_tools" class="mt-0.5 accent-yellow-400" />
            <span class="font-mono">{{ name }}</span>
          </label>

          <div v-if="otherAgents.length" class="text-[11px] text-gray-400 mt-2 mb-1">Colaboradores (delegate_agent):</div>
          <label v-for="a in otherAgents" :key="a.id" class="flex items-center gap-1.5 text-[11px] text-gray-400 mb-1 cursor-pointer">
            <input type="checkbox" :value="a.id" v-model="form.collaborator_ids" class="accent-yellow-400" />
            {{ a.name }}
          </label>

          <div v-if="store.skills.length" class="text-[11px] text-gray-400 mt-2 mb-1">Skills:</div>
          <label v-for="s in store.skills" :key="s.id" class="flex items-center gap-1.5 text-[11px] text-gray-400 mb-1 cursor-pointer" :title="s.description">
            <input type="checkbox" :value="s.id" v-model="form.enabled_skill_ids" class="accent-yellow-400" />
            {{ s.name }} <span class="text-gray-500">v{{ s.version }}</span>
          </label>

          <label class="flex items-center gap-1.5 text-[11px] text-gray-400 mt-2 cursor-pointer" title="Agente pode propor skills novas (exige sua aprovação)">
            <input type="checkbox" v-model="form.auto_learn" class="accent-yellow-400" />
            Auto-aprendizado (propõe skills)
          <span class="sr-only"> — agente pode propor skills novas, exige sua aprovação</span></label>
        </div>
        <div class="flex gap-2">
          <button @click="save" class="flex-1 px-2 py-1.5 text-xs rounded-md bg-accent-yellow/15 text-accent-yellow hover:bg-accent-yellow/25 transition-colors font-medium">Salvar</button>
          <button @click="editing = false" class="px-2 py-1.5 text-xs rounded-md border border-surface-600 text-gray-400">Cancelar</button>
        </div>
      </div>

      <!-- Propostas pendentes (auto-learn) -->
      <div v-if="store.proposals.length" class="card p-3 space-y-2 border-accent-yellow/40">
        <div class="text-xs font-semibold text-accent-yellow inline-flex items-center gap-1.5"><Icon name="raio" class="w-3.5 h-3.5" />Propostas de skill pendentes</div>
        <div v-for="p in store.proposals" :key="p.id" class="bg-surface-900 rounded-lg p-2.5">
          <div class="text-[11px] font-semibold text-gray-200">{{ p.action }} · {{ p.name }}</div>
          <div class="text-[11px] text-gray-400 mt-0.5">{{ p.rationale }}</div>
          <details class="mt-1">
            <summary class="text-[11px] text-gray-500 cursor-pointer">ver conteúdo</summary>
            <pre class="text-[11px] text-gray-400 mt-1 whitespace-pre-wrap max-h-40 overflow-y-auto">{{ p.content }}</pre>
          </details>
          <div class="flex gap-2 mt-1.5">
            <button @click="store.reviewProposal(p.id, 'approve')" class="text-[11px] px-2 py-0.5 rounded bg-accent-yellow/15 text-accent-yellow">Aprovar</button>
            <button @click="store.reviewProposal(p.id, 'reject')" class="text-[11px] px-2 py-0.5 rounded border border-surface-600 text-gray-400">Rejeitar</button>
          </div>
        </div>
      </div>

      <!-- Skills -->
      <div class="card p-3 space-y-2">
        <div class="flex items-center justify-between">
          <div class="text-xs font-semibold text-gray-200">Skills</div>
          <button @click="startNewSkill" class="text-[11px] text-gray-400 hover:text-accent-yellow">+ nova</button>
        </div>
        <div v-for="s in store.skills" :key="s.id" class="flex items-center justify-between text-[11px]">
          <span class="text-gray-400" :title="s.description">{{ s.name }} <span class="text-gray-500">v{{ s.version }}</span></span>
          <span class="flex gap-2">
            <button @click="editSkill(s)" class="text-gray-500 hover:text-gray-300">editar</button>
            <button @click="store.removeSkill(s.id)" class="text-gray-500 hover:text-accent-red-light">×</button>
          </span>
        </div>
        <div v-if="skillEditing" class="space-y-2 pt-1 border-t border-surface-700">
          <input v-model="skillForm.name" placeholder="Nome" class="inp" />
          <input v-model="skillForm.description" placeholder="Descrição curta" class="inp" />
          <textarea v-model="skillForm.content" placeholder="Procedimento (injetado no system prompt)" rows="4" class="inp resize-y" />
          <div class="flex gap-2">
            <button @click="saveSkill" class="flex-1 px-2 py-1 text-[11px] rounded bg-accent-yellow/15 text-accent-yellow">Salvar skill</button>
            <button @click="skillEditing = false" class="px-2 py-1 text-[11px] rounded border border-surface-600 text-gray-400">Cancelar</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Runner -->
    <div class="flex-1 flex flex-col overflow-hidden">
      <div v-if="!store.selected" class="flex-1 flex items-center justify-center text-center">
        <div>
          <p class="text-gray-400 text-sm">Selecione ou crie um agente</p>
          <p class="text-gray-500 text-xs mt-2 max-w-md">
            Agentes nativos rodam o loop de tools do basanos (memecoins, hype radar, busca social).<br/>
            Tipos hermes/openclaw delegam ao gateway configurado no .env.
          </p>
        </div>
      </div>

      <template v-else>
        <!-- Stream -->
        <div ref="streamEl" class="flex-1 overflow-y-auto p-4 space-y-3">
          <div v-for="(ev, i) in store.events" :key="i">
            <!-- user -->
            <div v-if="ev.type === 'user'" class="flex justify-end">
              <div class="max-w-2xl bg-accent-yellow/10 border border-accent-yellow/20 rounded-xl px-4 py-2.5 text-sm text-gray-100 whitespace-pre-wrap">{{ ev.content }}</div>
            </div>
            <!-- assistant -->
            <div v-else-if="ev.type === 'assistant'" class="flex" :class="{ 'pl-8': ev.depth > 0 }">
              <div class="max-w-3xl bg-surface-800 border border-surface-600 rounded-xl px-4 py-2.5 text-sm text-gray-200 whitespace-pre-wrap">
                <div v-if="ev.depth > 0" class="text-[11px] text-accent-yellow/70 mb-1">↳ {{ ev.agent }}</div>
                {{ ev.content }}
              </div>
            </div>
            <!-- tool call/result -->
            <div v-else class="max-w-3xl">
              <details class="bg-surface-900 border border-surface-700 rounded-lg px-3 py-2">
                <summary class="text-[11px] cursor-pointer" :class="ev.type === 'tool_call' ? 'text-accent-yellow/70' : 'text-gray-400'">
                  <Icon v-if="ev.type === 'tool_call'" name="engrenagem" class="w-3 h-3 mr-1" />{{ ev.type === 'tool_call' ? ev.name + '(' + JSON.stringify(ev.args) + ')' : '↳ resultado de ' + ev.name }}
                </summary>
                <pre v-if="ev.result" class="text-[11px] text-gray-400 mt-1.5 whitespace-pre-wrap max-h-48 overflow-y-auto">{{ ev.result }}</pre>
              </details>
            </div>
          </div>

          <div v-if="store.running" class="flex items-center gap-2 text-xs text-accent-yellow/70">
            <span class="dollar-loader-sm">$</span> Agente trabalhando...
          </div>
          <div v-if="store.error" class="card p-3 border-accent-red text-accent-red-light text-sm">{{ store.error }}</div>
        </div>

        <!-- Input -->
        <div class="p-4 border-t border-surface-500">
          <form @submit.prevent="send" class="flex gap-2">
            <input
              v-model="prompt"
              :placeholder="`Pergunte ao ${store.selected.name}... (ex: 'quais memecoins da robinhood têm hype real agora?')`"
              class="inp flex-1"
              :disabled="store.running"
            />
            <button type="submit" :disabled="store.running || !prompt.trim()"
                    class="px-4 py-2 text-sm rounded-lg bg-accent-yellow/15 text-accent-yellow hover:bg-accent-yellow/25 disabled:opacity-40 transition-colors font-medium">
              Enviar
            </button>
          </form>
        </div>
      </template>
    </div>
  </div>
</template>

<script setup>
import Icon from '@/components/common/Icon.vue'
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue'
import { useAgentsStore } from '@/stores/agents.js'

const store = useAgentsStore()
const editing = ref(false)
const skillEditing = ref(false)
const prompt = ref('')
const streamEl = ref(null)

const form = reactive({
  id: null, name: '', system_prompt: '', model: 'anthropic/claude-sonnet-5',
  temperature: 0.7, agent_type: 'native', base_url: '', enabled_tools: [],
  collaborator_ids: [], enabled_skill_ids: [], auto_learn: false,
})

const skillForm = reactive({ id: null, name: '', description: '', content: '' })

const otherAgents = computed(() => store.agents.filter((a) => a.id !== form.id))

function startNewSkill() {
  Object.assign(skillForm, { id: null, name: '', description: '', content: '' })
  skillEditing.value = true
}

function editSkill(s) {
  Object.assign(skillForm, s)
  skillEditing.value = true
}

async function saveSkill() {
  await store.saveSkill({ ...skillForm })
  skillEditing.value = false
}

onMounted(() => store.fetchAgents())

watch(() => store.events.length, () => {
  nextTick(() => { if (streamEl.value) streamEl.value.scrollTop = streamEl.value.scrollHeight })
})

function startNew() {
  Object.assign(form, {
    id: null,
    name: 'Caçador de Hype',
    system_prompt: 'Você é um analista degen de memecoins. Use as tools para buscar dados on-chain e sociais antes de opinar. Seja direto, cite números e sempre avise sobre riscos de rugpull (liquidez baixa, pool novo, hype fabricado por boosts).',
    model: 'anthropic/claude-sonnet-5',
    temperature: 0.7, agent_type: 'native', base_url: '',
    enabled_tools: ['degen_tokens', 'degen_hype', 'social_search'],
    collaborator_ids: [], enabled_skill_ids: [], auto_learn: false,
  })
  editing.value = true
}

function editAgent(a) {
  Object.assign(form, JSON.parse(JSON.stringify(a)))
  editing.value = true
}

async function save() {
  await store.saveAgent({ ...form })
  editing.value = false
}

function send() {
  const p = prompt.value.trim()
  if (!p) return
  prompt.value = ''
  store.run(p)
}
</script>

<style scoped>
.inp {
  @apply w-full bg-surface-900 border border-surface-600 rounded-lg px-3 py-2 text-sm text-gray-200
         placeholder-gray-600 focus:outline-none focus:border-accent-yellow/50;
}
</style>
