<script setup>
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from 'vue'

const stageNames = {
  queued: '排队中', reading: '读取文档', extracting: '抽取实体关系',
  graph: '构建知识图谱', traversing: '遍历语义子图', generating: '生成问答',
  exporting: '导出数据集', completed: '处理完成', failed: '运行失败',
}
const typeNames = {
  single_choice: '单选', multiple_choice: '多选', cot: '推理',
  true_false: '判断', fill_blank: '填空',
}
const stageMilestones = computed(() => (job.value?.mode || config.mode) === 'generate'
  ? ['载入图谱', '遍历图谱', '问答生成', '数据导出']
  : (job.value?.mode || config.mode) === 'build' ? ['读取文档', '知识抽取', '图谱构建']
    : ['读取文档', '知识抽取', '图谱构建', '问答生成', '数据导出'])
const config = reactive({
  mode: 'run', storage: 'json', graph_action: 'new', graph_id: null, graph_name: 'MilKG',
  neo4j_uri: 'bolt://127.0.0.1:7687', neo4j_user: 'neo4j', neo4j_database: 'neo4j',
  api_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  extract_model: 'qwen3.5-flash', generate_model: 'qwen3.5-plus',
  extract_temperature: 0.1, generate_temperature: 0.7,
  max_chars: 1800, overlap: 180, min_confidence: 0,
  max_subgraphs: 24, max_atomic: 24, per_subgraph: 1,
  include_atomic: true, materialize_specs: true,
  question_types: Object.keys(typeNames), output_format: 'alpaca', seed: 42,
})
const apiKey = ref('')
const neo4jPassword = ref('')
const graphList = ref([])
const graphListLoading = ref(false)
const graphListError = ref('')
const graphMigrationMessage = ref('')
const graphMigrationLoading = ref(false)
const showKey = ref(false)
const file = ref(null)
const fileInput = ref(null)
const dragging = ref(false)
const advanced = ref(false)
const health = ref(null)
const healthError = ref('')
const job = ref(null)
const requestError = ref('')
const submitting = ref(false)
const items = ref([])
const itemTotal = ref(0)
const page = ref(0)
const pageSize = 8
const selectedType = ref('all')
const expanded = ref(-1)
const logPanel = ref(null)
let timer = null

const isBusy = computed(() => submitting.value || ['queued', 'running'].includes(job.value?.status))
const canStart = computed(() => (config.mode === 'generate' || !!file.value) && !!health.value?.ok &&
  !isBusy.value && (config.mode === 'build' || config.question_types.length > 0) &&
  (config.storage !== 'neo4j' || (config.graph_action === 'new' && config.mode !== 'generate') || !!config.graph_id))
const currentStageIndex = computed(() => {
  if (!job.value) return -1
  if (job.value.status === 'completed') return stageMilestones.value.length
  if (job.value.mode === 'generate') {
    if (job.value.stage === 'graph') return 0
    if (job.value.stage === 'traversing') return 1
    if (job.value.stage === 'generating') return 2
    if (job.value.stage === 'exporting') return 3
    return -1
  }
  if (job.value.stage === 'reading') return 0
  if (job.value.stage === 'extracting') return 1
  if (job.value.stage === 'graph' || job.value.stage === 'traversing') return 2
  if (job.value.stage === 'generating') return 3
  if (job.value.stage === 'exporting') return 4
  return -1
})
const progressLabel = computed(() => job.value?.total ? `${job.value.current} / ${job.value.total}` : '— / —')

async function loadGraphs() {
  graphListLoading.value = true
  graphListError.value = ''
  try {
    const result = await api('/api/graphs/list', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uri: config.neo4j_uri, user: config.neo4j_user,
        password: neo4jPassword.value, database: config.neo4j_database }) })
    graphList.value = result.graphs
    if (!graphList.value.some(graph => graph.id === config.graph_id)) config.graph_id = null
  } catch (error) {
    graphListError.value = error.message
  } finally {
    graphListLoading.value = false
  }
}

async function migrateGraph() {
  if (!config.graph_id) return
  graphMigrationLoading.value = true
  graphMigrationMessage.value = ''
  graphListError.value = ''
  try {
    const result = await api('/api/graphs/migrate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ uri: config.neo4j_uri, user: config.neo4j_user,
        password: neo4jPassword.value, database: config.neo4j_database, graph_id: config.graph_id }) })
    graphMigrationMessage.value = `分类已更新：${result.tagged_nodes} 个实体标签，${result.converted_relations} 条关系类型。请刷新 Neo4j Browser。`
  } catch (error) {
    graphListError.value = error.message
  } finally {
    graphMigrationLoading.value = false
  }
}

function chooseFile(candidate) {
  if (!candidate) return
  const extension = candidate.name.split('.').pop()?.toLowerCase()
  if (!['txt', 'md', 'pdf', 'docx'].includes(extension)) {
    requestError.value = '仅支持 TXT、MD、PDF、DOCX 文件。'
    return
  }
  if (candidate.size > 20 * 1024 * 1024) {
    requestError.value = '文件不能超过 20 MB。'
    return
  }
  file.value = candidate
  requestError.value = ''
}
function onDrop(event) {
  dragging.value = false
  chooseFile(event.dataTransfer?.files?.[0])
}
function formatSize(size) {
  return size < 1024 * 1024 ? `${(size / 1024).toFixed(1)} KB` : `${(size / 1024 / 1024).toFixed(1)} MB`
}
function errorMessage(value) {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(x => x.msg || JSON.stringify(x)).join('；')
  return '请求失败，请查看后端日志。'
}
async function api(path, options) {
  const response = await fetch(path, options)
  if (!response.ok) {
    let detail = null
    try { detail = (await response.json()).detail } catch { /* server may return text */ }
    throw new Error(errorMessage(detail) || `HTTP ${response.status}`)
  }
  return response.json()
}
async function checkHealth() {
  try {
    health.value = await api('/api/health')
    healthError.value = ''
  } catch {
    health.value = null
    healthError.value = '后端未连接。请启动 milkg-web。'
  }
}
async function loadItems() {
  if (!job.value?.id) return
  try {
    const typeQuery = selectedType.value === 'all' ? '' : `&type=${encodeURIComponent(selectedType.value)}`
    const result = await api(`/api/jobs/${job.value.id}/items?offset=${page.value * pageSize}&limit=${pageSize}${typeQuery}`)
    items.value = result.items
    itemTotal.value = result.total
  } catch (error) {
    requestError.value = error.message
  }
}
async function refreshJob() {
  if (!job.value?.id) return
  try {
    const snapshot = await api(`/api/jobs/${job.value.id}`)
    const countChanged = snapshot.item_count !== job.value.item_count
    job.value = snapshot
    if (countChanged || snapshot.status === 'completed') await loadItems()
    if (!['queued', 'running'].includes(snapshot.status)) stopPolling()
  } catch (error) {
    requestError.value = error.message
    stopPolling()
  }
}
function stopPolling() {
  if (timer) clearInterval(timer)
  timer = null
}
function startPolling() {
  stopPolling()
  refreshJob()
  timer = setInterval(refreshJob, 1400)
}
async function startJob() {
  if (!canStart.value) return
  requestError.value = ''
  if (config.mode !== 'generate' && Number(config.overlap) >= Number(config.max_chars)) {
    requestError.value = '分块重叠量必须小于块长。'
    return
  }
  submitting.value = true
  try {
    const data = new FormData()
    if (config.mode !== 'generate') data.append('file', file.value)
    data.append('config', JSON.stringify(config))
    data.append('api_key', apiKey.value.trim())
    data.append('neo4j_password', neo4jPassword.value)
    const created = await api('/api/jobs', { method: 'POST', body: data })
    apiKey.value = ''
    page.value = 0
    items.value = []
    itemTotal.value = 0
    expanded.value = -1
    job.value = { id: created.id, status: created.status, stage: 'queued', progress: 0, logs: [], item_count: 0 }
    sessionStorage.setItem('milkg_job_id', created.id)
    startPolling()
  } catch (error) {
    requestError.value = error.message
  } finally {
    submitting.value = false
  }
}
function download(format = config.output_format) {
  if (job.value?.status !== 'completed') return
  window.location.href = `/api/jobs/${job.value.id}/download?format=${encodeURIComponent(format)}`
}
function displayAnswer(answer) {
  return Array.isArray(answer) ? answer.join('、') : answer
}
function timeOnly(value) {
  return value ? new Date(value).toLocaleTimeString('zh-CN', { hour12: false }) : ''
}
watch(page, loadItems)
watch(() => config.mode, mode => { if (mode === 'generate') { config.storage = 'neo4j'; config.graph_action = 'extend' } })
watch(() => config.storage, storage => { if (storage === 'json') { config.graph_action = 'new'; config.graph_id = null } })
watch(selectedType, () => { if (page.value) page.value = 0; else loadItems(); expanded.value = -1 })
watch(() => job.value?.logs?.length, async () => {
  await nextTick()
  if (logPanel.value) logPanel.value.scrollTop = logPanel.value.scrollHeight
})
onMounted(async () => {
  await checkHealth()
  const id = sessionStorage.getItem('milkg_job_id')
  if (id) {
    job.value = { id, item_count: -1 }
    await refreshJob()
    if (['queued', 'running'].includes(job.value?.status)) startPolling()
  }
})
onUnmounted(stopPolling)
</script>

<template>
  <div class="page-shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark">M<span>·</span></span><span>MilKG <small>DATA STUDIO</small></span></div>
      <div class="topbar-right"><span class="topbar-index">KNOWLEDGE → DATA</span><span class="connection" :class="health?.ok ? 'online' : 'offline'"><i></i>{{ health?.ok ? '本地服务已连接' : '服务未连接' }}</span></div>
    </header>

    <main>
      <section class="intro">
        <div class="eyebrow"><span class="eyebrow-line"></span> DOCUMENT · GRAPH · DATASET <span class="version">/ 01</span></div>
        <h1>让多份文档的知识，<br /><em>汇入同一张图谱。</em></h1>
        <p>按需构图、扩展已有图谱，或从图谱独立生成 SFT 问答。每一步都清晰可见。</p>
      </section>

      <div v-if="healthError" class="service-alert"><span>!</span>{{ healthError }}<button @click="checkHealth">重新连接 ↗</button></div>

      <section class="workspace">
        <div class="panel config-panel">
          <div class="panel-heading"><div><span class="section-no">01 / INPUT</span><h2>数据与参数</h2></div><span class="heading-icon">↗</span></div>

          <div class="field-block"><div class="field-head"><label>运行方式</label><span>选择本次任务</span></div>
            <div class="mode-selector">
              <label :class="{ selected: config.mode === 'run' }"><input v-model="config.mode" type="radio" value="run" /><strong>构图并生成</strong><small>上传文档，完成全流程</small></label>
              <label :class="{ selected: config.mode === 'build' }"><input v-model="config.mode" type="radio" value="build" /><strong>只构建图谱</strong><small>抽取实体关系并保存</small></label>
              <label :class="{ selected: config.mode === 'generate' }"><input v-model="config.mode" type="radio" value="generate" /><strong>只生成 SFT</strong><small>读取已有 Neo4j 图谱</small></label>
            </div>
          </div>

          <div v-if="config.mode !== 'generate'" class="field-block">
            <div class="field-head"><label>原始文档</label><span>≤ 20 MB</span></div>
            <div class="upload-zone" :class="{ dragging, filled: file }" tabindex="0" role="button" aria-label="上传文档" @click="fileInput?.click()" @keydown.enter="fileInput?.click()" @dragover.prevent="dragging = true" @dragleave.prevent="dragging = false" @drop.prevent="onDrop">
              <input ref="fileInput" type="file" accept=".txt,.md,.pdf,.docx" hidden @change="chooseFile($event.target.files?.[0])" />
              <template v-if="file"><span class="file-icon">TXT</span><span class="file-detail"><strong>{{ file.name }}</strong><small>{{ formatSize(file.size) }} · 点击更换文件</small></span><span class="upload-arrow">↗</span></template>
              <template v-else><span class="upload-symbol">↥</span><strong>点击选择或拖入文档</strong><small>支持 TXT / MD / PDF / DOCX</small></template>
            </div>
          </div>

          <div class="divider"></div>
          <div class="field-block"><div class="field-head"><label>图谱存储</label><span>{{ config.storage === 'neo4j' ? '跨任务共享' : '任务本地文件' }}</span></div>
            <div class="storage-selector"><label v-if="config.mode !== 'generate'"><input v-model="config.storage" type="radio" value="json" />本地 JSON</label><label><input v-model="config.storage" type="radio" value="neo4j" />Neo4j</label></div>
            <template v-if="config.storage === 'neo4j'">
              <div class="neo4j-grid"><div><label class="sub-label" for="neo4j-uri">连接地址</label><input id="neo4j-uri" v-model.trim="config.neo4j_uri" placeholder="bolt://127.0.0.1:7687" /></div><div><label class="sub-label" for="neo4j-database">数据库</label><input id="neo4j-database" v-model.trim="config.neo4j_database" placeholder="neo4j" /></div></div>
              <div class="neo4j-grid"><div><label class="sub-label" for="neo4j-user">用户名</label><input id="neo4j-user" v-model.trim="config.neo4j_user" placeholder="neo4j" /></div><div><label class="sub-label" for="neo4j-password">密码 <span class="optional">留空读取 MILKG_NEO4J_PASSWORD</span></label><input id="neo4j-password" v-model="neo4jPassword" type="password" autocomplete="off" /></div></div>
              <div v-if="config.mode !== 'generate'" class="storage-selector graph-action"><label><input v-model="config.graph_action" type="radio" value="new" />新建逻辑图谱</label><label><input v-model="config.graph_action" type="radio" value="extend" />扩展已有图谱</label></div>
              <div v-if="config.graph_action === 'new' && config.mode !== 'generate'"><label class="sub-label" for="graph-name">新图谱名称</label><input id="graph-name" v-model.trim="config.graph_name" placeholder="例如：轻武器资料库" /></div>
              <div v-else><div class="graph-list-head"><label class="sub-label" for="graph-id">已有图谱</label><button type="button" :disabled="graphListLoading" @click="loadGraphs">{{ graphListLoading ? '读取中…' : '读取图谱列表 ↗' }}</button></div><select id="graph-id" v-model="config.graph_id" class="graph-select"><option :value="null">请选择图谱</option><option v-for="graph in graphList" :key="graph.id" :value="graph.id">{{ graph.name }} · {{ graph.node_count }} 节点 / {{ graph.edge_count }} 关系</option></select><button type="button" :disabled="!config.graph_id || graphMigrationLoading || isBusy" @click="migrateGraph">{{ graphMigrationLoading ? '更新分类中…' : '更新旧图谱的实体与关系分类' }}</button><p v-if="graphMigrationMessage" class="field-hint">{{ graphMigrationMessage }}</p><p v-if="graphListError" class="form-error">{{ graphListError }}</p></div>
              <span class="field-hint">密码仅用于本次连接。新建图谱不会清空已有数据；扩展会合并实体及来源。</span>
            </template>
          </div>

          <div class="divider"></div>
          <div class="field-block"><div class="field-head"><label>API 接入</label><span>阿里云百炼 / Qwen</span></div>
            <label class="sub-label" for="api-url">服务地址</label><input id="api-url" v-model.trim="config.api_url" type="url" spellcheck="false" />
            <label class="sub-label" for="api-key">API Key <span class="optional">留空读取 ALIYUN_API_KEY</span></label>
            <div class="key-wrap"><input id="api-key" v-model="apiKey" :type="showKey ? 'text' : 'password'" autocomplete="off" placeholder="sk-... 或使用环境变量" /><button type="button" @click="showKey = !showKey">{{ showKey ? '隐藏' : '显示' }}</button></div>
            <span class="field-hint">仅用于当前任务请求；界面不会保存密钥。</span>
          </div>

          <div class="divider"></div>
          <div class="field-block"><div class="field-head"><label>模型配置</label><span>按任务启用</span></div>
            <div class="model-grid" :class="{ single: config.mode !== 'run' }"><div v-if="config.mode !== 'generate'"><label class="sub-label" for="extract-model">实体关系提取</label><input id="extract-model" v-model.trim="config.extract_model" list="extract-models" placeholder="qwen3.5-flash" /><datalist id="extract-models"><option value="qwen3.5-flash" /><option value="qwen-plus" /><option value="qwen-turbo" /></datalist></div><div v-if="config.mode !== 'build'"><label class="sub-label" for="generate-model">问答合成</label><input id="generate-model" v-model.trim="config.generate_model" list="generate-models" placeholder="qwen3.5-plus" /><datalist id="generate-models"><option value="qwen3.5-plus" /><option value="qwen-max" /><option value="qwen-plus" /></datalist></div></div>
            <div v-if="config.mode !== 'generate'" class="temp-row"><label for="extract-temp">提取温度 <b>{{ Number(config.extract_temperature).toFixed(1) }}</b></label><input id="extract-temp" v-model.number="config.extract_temperature" type="range" min="0" max="1.5" step="0.1" /></div>
            <div v-if="config.mode !== 'build'" class="temp-row"><label for="generate-temp">合成温度 <b>{{ Number(config.generate_temperature).toFixed(1) }}</b></label><input id="generate-temp" v-model.number="config.generate_temperature" type="range" min="0" max="1.5" step="0.1" /></div>
          </div>

          <button class="advanced-toggle" type="button" :aria-expanded="advanced" @click="advanced = !advanced"><span>高级参数</span><span>{{ advanced ? '−' : '+' }}</span></button>
          <div v-if="advanced" class="advanced-content">
            <div class="number-grid"><label v-if="config.mode !== 'generate'">分块字符数<input v-model.number="config.max_chars" type="number" min="400" max="8000" /></label><label v-if="config.mode !== 'generate'">重叠字符数<input v-model.number="config.overlap" type="number" min="0" max="1000" /></label><label v-if="config.mode !== 'build'">最多语义子图<input v-model.number="config.max_subgraphs" type="number" min="1" max="200" /></label><label v-if="config.mode !== 'build'">最多单跳事实<input v-model.number="config.max_atomic" type="number" min="0" max="100" /></label><label v-if="config.mode !== 'build'">每图生成次数<input v-model.number="config.per_subgraph" type="number" min="1" max="5" /></label><label v-if="config.mode !== 'generate'">最低置信度<input v-model.number="config.min_confidence" type="number" min="0" max="1" step="0.05" /></label></div>
            <p v-if="config.mode !== 'build'" class="advanced-hint">语义子图上限只限制多关系出题素材的数量；实际数量取决于图谱中的关系。单跳事实和技术规格需要在下方分别开启。</p>
            <div v-if="config.mode !== 'build'" class="advanced-line"><span>问答题型</span><div class="check-grid"><label v-for="(name, type) in typeNames" :key="type" class="check-option"><input v-model="config.question_types" type="checkbox" :value="type" />{{ name }}</label></div></div>
            <div class="switch-line"><label v-if="config.mode !== 'build'"><input v-model="config.include_atomic" type="checkbox" />加入单跳事实</label><label v-if="config.mode !== 'generate'"><input v-model="config.materialize_specs" type="checkbox" />补充技术规格关系</label></div>
          </div>
          <div class="start-area"><button class="primary-button" :disabled="!canStart" @click="startJob"><span>{{ isBusy ? '任务运行中' : config.mode === 'build' ? '开始构建图谱' : config.mode === 'generate' ? '从图谱生成 SFT' : '开始生成数据集' }}</span><span>{{ isBusy ? '···' : '→' }}</span></button><p v-if="requestError" class="form-error">{{ requestError }}</p><p v-else>任务进度和结果会在右侧实时更新。</p></div>
        </div>

        <div class="right-column">
          <div class="panel progress-panel"><div class="panel-heading"><div><span class="section-no">02 / PROCESS</span><h2>运行进度</h2></div><span class="status-chip" :class="job?.status || 'idle'"><i></i>{{ job ? (job.status === 'completed' ? '已完成' : job.status === 'failed' ? '失败' : '运行中') : '等待开始' }}</span></div>
            <div class="progress-display"><div><span class="progress-caption">CURRENT STAGE</span><h3>{{ job ? stageNames[job.stage] || '处理中' : '准备就绪' }}</h3><p>{{ job?.status === 'failed' ? job.error : job ? `${job.filename || '文档'} · ${progressLabel}` : '上传文档并设置参数，开始构建数据集。' }}</p></div><div class="progress-number">{{ job?.progress ?? 0 }}<small>%</small></div></div>
            <div class="progress-track"><div :style="{ width: `${job?.progress ?? 0}%` }"></div></div>
            <div class="milestones"><div v-for="(name, index) in stageMilestones" :key="name" :class="{ reached: currentStageIndex >= index, active: currentStageIndex === index }"><i></i><span>{{ name }}</span></div></div>
            <div v-if="job?.graph_nodes" class="graph-metrics"><span><b>{{ job.graph_nodes }}</b> 节点</span><span><b>{{ job.graph_edges }}</b> 关系边</span><span><b>{{ job.semantic_subgraphs }}</b> 语义子图</span><span><b>{{ job.atomic_subgraphs }}</b> 单跳事实</span></div><p v-if="job?.graph_id" class="graph-id">Neo4j 图谱 ID：{{ job.graph_id }}</p>
          </div>

          <div class="panel log-panel"><div class="panel-heading"><div><span class="section-no">03 / ACTIVITY</span><h2>运行日志</h2></div><span class="log-count">{{ job?.logs?.length || 0 }} EVENTS</span></div>
            <div ref="logPanel" class="terminal"><div class="terminal-bar"><span class="terminal-dots"><i></i><i></i><i></i></span><span>milkg / process.log</span><span>● LIVE</span></div><div class="terminal-body"><div v-if="!job?.logs?.length" class="terminal-empty"><span>▍</span>等待任务启动，日志将显示在这里。</div><div v-for="(entry, index) in job?.logs || []" :key="index" class="log-line" :class="entry.level"><time>{{ timeOnly(entry.time) }}</time><span class="log-symbol">{{ entry.level === 'error' ? '×' : entry.level === 'warning' ? '!' : '›' }}</span><span>{{ entry.message }}</span></div></div></div>
          </div>
        </div>
      </section>

      <section v-if="job?.mode === 'build' || (!job && config.mode === 'build')" class="preview-section"><div class="preview-title"><div><span class="section-no">04 / OUTPUT</span><h2>知识图谱</h2><p>构图任务完成后，可切换到“只生成 SFT”读取图谱。</p></div></div><div class="empty-preview"><div class="empty-graphic"><span>DOC</span><span>→</span><span>KG</span></div><h3>{{ job?.status === 'completed' ? '图谱已保存' : '等待构图' }}</h3><p v-if="job?.graph_id">图谱 ID：{{ job.graph_id }}</p><p v-else>Neo4j 模式可在下一次任务中继续扩展图谱。</p></div></section>
      <section v-else class="preview-section"><div class="preview-title"><div><span class="section-no">04 / OUTPUT</span><h2>数据集预览 <span>{{ itemTotal ? String(itemTotal).padStart(2, '0') : '—' }}</span></h2><p>逐条查看问答内容、内部证据编号与生成策略。</p></div><div class="export-controls"><select v-model="config.output_format" aria-label="导出格式"><option value="alpaca">Alpaca JSON</option><option value="sharegpt">ShareGPT JSON</option><option value="chatml">ChatML JSON</option></select><button :disabled="job?.status !== 'completed'" @click="download()">下载数据集 <span>↗</span></button></div></div>
        <div v-if="job?.item_count" class="preview-card"><div class="preview-toolbar"><div class="filter-tabs"><button v-for="(name, type) in { all: '全部', ...typeNames }" :key="type" :class="{ selected: selectedType === type }" @click="selectedType = type">{{ name }}</button></div><span>{{ itemTotal ? `第 ${page * pageSize + 1}–${Math.min(itemTotal, (page + 1) * pageSize)} 条` : '无结果' }} / 共 {{ itemTotal }} 条</span></div>
          <div v-if="!items.length" class="filter-empty">没有该题型的问答。请选择其他题型。</div>
          <article v-for="(item, index) in items" :key="`${page}-${index}`" class="qa-item"><button class="qa-summary" @click="expanded = expanded === index ? -1 : index"><span class="qa-index">{{ String(page * pageSize + index + 1).padStart(2, '0') }}</span><span class="qa-main"><span class="qa-tags"><span>{{ typeNames[item.type] || item.type }}</span><span>{{ item.difficulty === 'easy' ? '基础' : item.difficulty === 'hard' ? '进阶' : '中等' }}</span></span><strong>{{ item.question }}</strong><small>答案：{{ displayAnswer(item.answer) }}</small></span><span class="expand-mark">{{ expanded === index ? '−' : '+' }}</span></button><div v-if="expanded === index" class="qa-detail"><div v-if="item.options?.length" class="option-grid"><div v-for="option in item.options" :key="option.label" :class="{ correct: item.answer?.includes(option.label) }"><b>{{ option.label }}</b>{{ option.text }}</div></div><div class="detail-row"><span>解释</span><p>{{ item.explanation || '—' }}</p></div><div class="detail-row"><span>支持事实</span><p>{{ item.supporting_facts?.join(' · ') || '—' }}</p></div><div class="detail-row"><span>生成策略</span><p>{{ item.strategy || '—' }}</p></div></div></article>
          <div v-if="itemTotal" class="pagination"><button :disabled="page === 0" @click="page--; expanded = -1">← 上一页</button><span>{{ page + 1 }} / {{ Math.ceil(itemTotal / pageSize) }}</span><button :disabled="(page + 1) * pageSize >= itemTotal" @click="page++; expanded = -1">下一页 →</button></div>
        </div>
        <div v-else class="empty-preview"><div class="empty-graphic"><span>01</span><span>→</span><span>KG</span><span>→</span><span>SFT</span></div><h3>结果会在这里逐条出现</h3><p>启动任务后，已通过校验的问答会实时进入预览区。</p></div>
      </section>
    </main>
    <footer><span>MilKG / Knowledge-grounded SFT pipeline</span><span>LOCAL WORKBENCH · 2026</span></footer>
  </div>
</template>
