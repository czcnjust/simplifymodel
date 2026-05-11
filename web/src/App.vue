<template>
  <div class="app">
    <!-- 左侧：Agent 对话 -->
    <ChatPanel
      :messages="messages"
      :status-type="statusType"
      :status-text="statusText"
      :is-sending="isSending"
      @send="handleSendMessage"
      @clear-chat="clearChat"
      @file-upload="handleFileUpload"
      ref="chatPanelRef"
    />

    <!-- 右侧：实验结果总览 -->
    <section class="panel">
      <div class="panel-header">
        <div class="panel-title">实验结果总览</div>
        <div class="badge">{{ rightBadgeText }}</div>
      </div>

      <div class="right-body">
        <!-- 模型指标 -->
        <MetricsComparison
          :current-metrics="currentModelMetrics"
        />

        <!-- 实验过程与产物 -->
        <ResultTabs
          ref="resultTabsRef"
          :current-tab="currentTab"
          :log-text="logText"
          :netron-url="netronUrl"
          :structure-text="structureText"
          :improved-code="improvedCode"
          :baseline-ready="baselineReady"
          :improved-ready="improvedReady"
          @tab-change="switchTab"
          @netron-open="openNetron"
          @model-select="handleModelSelect"
          @show-toast="showToastMessage"
        />
      </div>
    </section>

    <!-- Toast 提示 -->
    <Toast :message="toastMessage" :show="showToast" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'
import ChatPanel from './components/ChatPanel.vue'
import MetricsComparison from './components/MetricsComparison.vue'
import ResultTabs from './components/ResultTabs.vue'
import Toast from './components/Toast.vue'
import { APP_CONFIG } from './config.js'

// 状态变量
const messages = ref([
  {
    role: 'agent',
    text: APP_CONFIG.welcomeMessage,
  },
])
const isSending = ref(false)
const statusType = ref('')
const statusText = ref('当前状态：空闲')
const rightBadgeText = ref('等待实验')
const currentTab = ref('log')
const netronUrl = ref('')
const structureType = ref('baseline')  // 添加缺失的变量
const showToast = ref(false)
const toastMessage = ref('')
const logText = ref('')

const latestResult = ref(null)
const pollTimer = ref(null)
const currentModelMetrics = ref(null)  // 当前选中模型的 metrics

// ChatPanel 引用
const chatPanelRef = ref(null)
// ResultTabs 引用
const resultTabsRef = ref(null)

// 计算属性
const baselineReady = computed(
  () => !!(latestResult.value?.baseline_ready && latestResult.value?.baseline)
)
const improvedReady = computed(
  () => !!(latestResult.value?.improved_ready && latestResult.value?.improved)
)
const bothReady = computed(() => baselineReady.value && improvedReady.value)

const baseline = computed(() => latestResult.value?.baseline || {})
const improved = computed(() => latestResult.value?.improved || {})
const agentReport = computed(() => latestResult.value?.agent_report || '')
const improvedCode = computed(() => latestResult.value?.improved_code || '')

const structureText = computed(() => {
  if (!latestResult.value) return ''
  if (structureType.value === 'baseline') {
    return baselineReady.value
      ? latestResult.value.baseline_structure || '暂无 baseline 结构'
      : 'Baseline 尚未训练完成，暂无结构。'
  } else {
    return improvedReady.value
      ? latestResult.value.improved_structure || '暂无 improved 结构'
      : 'Improved 尚未训练完成，暂无结构。'
  }
})

const accChange = computed(() => {
  if (!bothReady.value) return 0
  return percentChange(baseline.value.accuracy, improved.value.accuracy)
})

const paramsChange = computed(() => {
  if (!bothReady.value) return 0
  return percentChange(baseline.value.params, improved.value.params)
})

const sizeChange = computed(() => {
  if (!bothReady.value) return 0
  return percentChange(baseline.value.model_size_mb, improved.value.model_size_mb)
})

// 计算百分比变化
function percentChange(oldValue, newValue) {
  oldValue = Number(oldValue || 0)
  newValue = Number(newValue || 0)

  if (oldValue === 0) {
    return 0
  }

  return ((newValue - oldValue) / oldValue) * 100
}

// API 请求函数
async function apiJson(url, options = {}) {
  const finalOptions = {
    cache: 'no-store',
    ...options,
  }

  try {
    const res = await axios.get(url, finalOptions)
    return res.data
  } catch (error) {
    throw new Error(error.response?.data?.message || error.message)
  }
}

async function apiPost(url, data) {
  try {
    const res = await axios.post(url, data)
    return res.data
  } catch (error) {
    throw new Error(error.response?.data?.message || error.message)
  }
}

// Toast 提示
function showToastMessage(text) {
  toastMessage.value = text
  showToast.value = true

  setTimeout(() => {
    showToast.value = false
  }, 3200)
}

// 处理文件上传
async function handleFileUpload(file, text) {
  const userMessage = '上传模型文件：' + file.name
  messages.value.push({
    role: 'user',
    text: userMessage,
  })

  const formData = new FormData()
  formData.append('file', file)

  try {
    const res = await axios.post('/api/model/upload', formData)
    const data = res.data

    if (data.success) {
      messages.value.push({
        role: 'agent',
        text: data.message || '已保存上传的模型文件。',
      })
      showToastMessage('已保存上传模型，等待训练')
      
      // 重置 ChatPanel 中的文件选择
      if (chatPanelRef.value) {
        chatPanelRef.value.fileNameHint = '未选择文件'
      }
      
      // 刷新所有数据
      await refreshAll()
      
      // 主动刷新模型列表（无论当前在哪个页签）
      console.log('[文件上传] resultTabsRef:', resultTabsRef.value)
      if (resultTabsRef.value) {
        console.log('[文件上传] 正在刷新模型列表...')
        resultTabsRef.value.loadGeneratedModels()
        console.log('[文件上传] 模型列表刷新完成')
      } else {
        console.warn('[文件上传] resultTabsRef 为空，无法刷新模型列表')
      }
    } else {
      messages.value.push({
        role: 'agent',
        text: data.message || '上传失败。',
      })
      setStatus('failed', '当前状态：上传失败')
    }
  } catch (e) {
    showToastMessage('上传失败：' + e.message)
  }
}

// 处理发送消息
async function handleSendMessage(text) {
  if (!text) {
    showToastMessage('请先输入消息、优化需求，或者上传 .py 文件。')
    return
  }

  isSending.value = true

  try {
    messages.value.push({ role: 'user', text })

    // 直接使用流式请求
    const agentMessageIndex = messages.value.length
    messages.value.push({
      role: 'agent',
      text: '',
      streaming: true  // 标记为流式消息
    })
    
    // 使用 fetch 进行流式接收
    await handleStreamChat(text, agentMessageIndex)
    
    setStatus('success', '当前状态：Agent 已处理消息')
    await refreshAll()
  } catch (e) {
    messages.value.push({
      role: 'agent',
      text: '请求失败：' + e.message,
    })
    setStatus('failed', '当前状态：请求失败')
  } finally {
    isSending.value = false
  }
}

// 处理流式聊天
async function handleStreamChat(text, messageIndex) {
  try {
    console.log('[流式请求] 开始发送消息:', text)
    
    const response = await fetch('/api/agent/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message: text, stream: true })
    })

    if (!response.ok) {
      throw new Error('网络响应错误')
    }

    // 检查 Content-Type
    const contentType = response.headers.get('content-type')
    console.log('[流式请求] Content-Type:', contentType)
    
    // 兼容多种 SSE Content-Type
    const isStreamResponse = contentType && (
      contentType.includes('text/event-stream') ||
      contentType.includes('application/jsonl') ||
      contentType.includes('text/plain')
    )
    
    if (!isStreamResponse) {
      // 非流式响应,按普通 JSON 处理
      console.log('[流式请求] 检测到非流式响应')
      const data = await response.json()
      console.log('[流式请求] 响应数据:', data)
      messages.value[messageIndex].text = data.message || 'Agent 已收到。'
      messages.value[messageIndex].streaming = false
      
      // 处理训练等特殊动作
      if (isTrainingAction(data)) {
        setStatus('running', '当前状态:训练任务已启动,正在等待后端日志...')
        rightBadgeText.value = '运行中'
        switchTab('log')
        logText.value = '训练任务已启动,正在等待后端返回训练日志...'
        startPolling()
      }
      return
    }

    console.log('[流式请求] 开始流式接收')
    // 流式响应处理
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let fullText = ''
    let chunkCount = 0
    let hasStartedTask = false  // 标记是否已启动任务

    while (true) {
      const { done, value } = await reader.read()
      
      if (done) {
        console.log('[流式请求] 流式接收完成,共接收', chunkCount, '个片段')
        // 流式完成后,将 Markdown 转换为 HTML
        messages.value[messageIndex].text = fullText
        messages.value[messageIndex].streaming = false
        break
      }

      chunkCount++
      const chunk = decoder.decode(value, { stream: true })
      
      const lines = chunk.split('\n')

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6))
            
            // 处理不同类型的流式数据
            if (data.type === 'thinking') {
              // 思考过程（特殊样式）
              console.log('[流式请求] 思考:', data.content)
              fullText += `<div class="thinking-block">💭 <strong>模型思考过程：</strong><br/>${data.content}</div>\n`
              messages.value[messageIndex].text = fullText
              scrollToBottom()
            } else if (data.type === 'status') {
              // 状态消息
              console.log('[流式请求] 状态:', data.message)
              fullText += data.message + '\n'
              messages.value[messageIndex].text = fullText
              scrollToBottom()
            } else if (data.type === 'log') {
              // 日志消息（训练、量化等任务的日志）
              console.log('[流式请求] 日志:', data.content)
              fullText += data.content + '\n'
              messages.value[messageIndex].text = fullText
              
              // 同时更新右侧日志面板
              logText.value = fullText
              
              // 如果是第一次收到日志，切换到日志标签并启动轮询
              if (!hasStartedTask) {
                hasStartedTask = true
                setStatus('running', '当前状态:任务运行中,正在等待后端日志...')
                rightBadgeText.value = '运行中'
                switchTab('log')
                startPolling()
              }
              
              scrollToBottom()
            } else if (data.content) {
              // 普通文本内容（chat 意图）
              fullText += data.content
              messages.value[messageIndex].text = fullText
              scrollToBottom()
            } else if (data.error) {
              // 错误消息
              messages.value[messageIndex].text = data.error
              messages.value[messageIndex].streaming = false
              console.log('[流式请求] 收到错误:', data.error)
              return
            } else if (data.done) {
              // 流式完成
              messages.value[messageIndex].streaming = false
              console.log('[流式请求] 收到完成信号')
              
              // 如果启动了任务，刷新结果
              if (hasStartedTask) {
                await refreshAll()
              }
              
              return
            }
          } catch (e) {
            console.error('解析流式数据失败:', e)
            console.error('原始数据行:', line)
            console.error('原始数据长度:', line.length)
            console.error('原始数据前100字符:', line.substring(0, 100))
          }
        }
      }
    }
  } catch (e) {
    console.error('流式请求失败:', e)
    messages.value[messageIndex].text = '流式对话失败:' + e.message
    messages.value[messageIndex].streaming = false
  }
}

// 滚动到底部
function scrollToBottom() {
  setTimeout(() => {
    const chatBody = document.getElementById('chatBody')
    if (chatBody) {
      chatBody.scrollTop = chatBody.scrollHeight
    }
  }, 0)
}

// 判断是否为训练动作
function isTrainingAction(data) {
  if (!data) return false

  const action = String(data.action || '').toLowerCase()
  const message = String(data.message || '').toLowerCase()
  const runType = String(data.run_type || '').toLowerCase()

  if (data.running === true) return true
  if (data.task_started === true) return true
  if (data.start_polling === true) return true

  if (action.includes('train')) return true

  if (action.includes('baseline') || action.includes('improved')) {
    if (
      message.includes('训练') ||
      message.includes('train') ||
      message.includes('启动') ||
      message.includes('started')
    ) {
      return true
    }
  }

  if (runType === 'baseline' || runType === 'improved') {
    if (
      message.includes('训练') ||
      message.includes('train') ||
      message.includes('启动') ||
      message.includes('started')
    ) {
      return true
    }
  }

  if (message.includes('训练任务已启动')) return true
  if (message.includes('已启动')) return true
  if (message.includes('开始训练')) return true
  if (message.includes('start') && message.includes('train')) return true

  return false
}

// 设置状态
function setStatus(type, text) {
  statusType.value = type
  statusText.value = text
}

// 轮询任务状态
async function startPolling() {
  if (pollTimer.value) {
    clearInterval(pollTimer.value)
    pollTimer.value = null
  }

  await refreshTaskStatus()

  pollTimer.value = setInterval(async () => {
    const status = await refreshTaskStatus()

    if (!status) return

    if (!status.running) {
      clearInterval(pollTimer.value)
      pollTimer.value = null

      await refreshAll()

      if (status.status === 'success') {
        messages.value.push({
          role: 'agent',
          text: '训练任务完成。我已经刷新了右侧的指标、结构图、生成代码和文件下载。',
        })
        setStatus('success', '当前状态：训练完成')
      } else if (status.status === 'failed') {
        messages.value.push({
          role: 'agent',
          text: status.message || '训练任务失败。',
        })
        setStatus('failed', '当前状态：训练失败')
      } else {
        setStatus('', '当前状态：空闲')
      }
    }
  }, 1000)
}

// 刷新任务状态
async function refreshTaskStatus() {
  try {
    const status = await apiJson('/api/task/status?t=' + Date.now())

    const logTextValue = normalizeTaskLogs(status.logs)

    if (logTextValue && logTextValue.trim()) {
      logText.value = logTextValue
    } else if (status.running) {
      logText.value = '训练任务已启动，正在等待后端返回训练日志...'
    } else {
      logText.value = '暂无日志'
    }

    if (status.running) {
      setStatus('running', '当前状态：' + (status.message || '任务运行中'))
      rightBadgeText.value = '运行中'
    } else if (status.status === 'success') {
      setStatus('success', '当前状态：' + (status.message || '任务完成'))
      rightBadgeText.value = '已完成'
    } else if (status.status === 'failed') {
      setStatus('failed', '当前状态：' + (status.message || '任务失败'))
      rightBadgeText.value = '任务失败'
    } else {
      setStatus('', '当前状态：空闲')
      rightBadgeText.value = '等待操作'
    }

    return status
  } catch (e) {
    logText.value = '读取任务状态失败：' + e.message
    return null
  }
}

// 标准化日志
function normalizeTaskLogs(logs) {
  if (typeof logs === 'string') {
    return logs
  }

  if (Array.isArray(logs)) {
    return logs.join('\n')
  }

  return ''
}

// 刷新所有数据
async function refreshAll() {
  await refreshTaskStatus()

  try {
    const result = await apiJson('/api/result?t=' + Date.now())
    latestResult.value = result
  } catch (e) {
    showToastMessage('刷新结果失败：' + e.message)
  }
}

// 清空对话界面
function clearChat() {
  // 保留第一条欢迎消息，清空其他消息
  messages.value = [
    {
      role: 'agent',
      text: APP_CONFIG.welcomeMessage,
    },
  ]
  showToastMessage('✓ 对话已清空')
}

// 切换标签页
function switchTab(name) {
  currentTab.value = name
  
  // 如果切换到 netron 标签且还没有 URL，不自动打开，等待用户选择模型
  // if (name === 'netron' && !netronUrl.value) {
  //   openNetron('baseline')
  // }
}

// 打开 Netron（新版本 - 直接接收 URL）
function openNetron(url) {
  netronUrl.value = url
}

// 处理模型选择事件
async function handleModelSelect(runType) {
  try {
    const res = await axios.get(`/api/runs/${runType}/metrics`)
    if (res.data.success) {
      currentModelMetrics.value = res.data.metrics
    } else {
      currentModelMetrics.value = null
    }
  } catch (e) {
    console.error('获取模型 metrics 失败:', e)
    currentModelMetrics.value = null
  }
}

// 初始化
onMounted(() => {
  refreshAll()
})
</script>

<style>
/* 全局 CSS 变量 */
:root {
  --bg: #edf2f7;
  --panel: #ffffff;
  --panel-soft: #f8fafc;
  --border: #dbe4f0;
  --border-strong: #c9d6e6;
  --text: #0f172a;
  --muted: #64748b;
  --blue: #2563eb;
  --blue-soft: #eff6ff;
  --green: #16a34a;
  --green-soft: #ecfdf5;
  --red: #dc2626;
  --red-soft: #fef2f2;
  --orange: #f97316;
  --orange-soft: #fff7ed;
  --purple: #7c3aed;
  --purple-soft: #f5f3ff;
  --dark: #0f172a;
  --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
  --radius-xl: 28px;
  --radius-lg: 18px;
  --radius-md: 14px;
}

/* 字体渲染优化 */
html {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

body {
  margin: 0;
  background: var(--bg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
  color: var(--text);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
</style>

<style scoped>
/* 主应用布局 */
.app {
  width: 98%;
  height: 95vh;
  padding: 22px;
  display: grid;
  grid-template-columns: minmax(520px, 1fr) minmax(520px, 1fr);
  gap: 22px;
  overflow: hidden;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

.panel {
  background: var(--panel);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow);
  overflow: hidden;
  min-height: calc(100vh - 44px);
  display: flex;
  flex-direction: column;
}

.panel-header {
  height: 86px;
  padding: 0 28px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}

.panel-title {
  font-size: 28px;
  font-weight: 900;
  letter-spacing: -0.8px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.badge {
  padding: 10px 18px;
  border-radius: 999px;
  background: var(--blue-soft);
  color: var(--blue);
  font-weight: 800;
  font-size: 15px;
}

/* 思考过程样式 */
.thinking-block {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 16px 20px;
  border-radius: 12px;
  margin: 12px 0;
  box-shadow: 0 4px 6px rgba(102, 126, 234, 0.2);
  line-height: 1.6;
}

.thinking-block strong {
  display: block;
  margin-bottom: 8px;
  font-size: 15px;
  opacity: 0.95;
}

.right-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 24px 28px 30px;
  display: flex;
  flex-direction: column;
}

/* 响应式设计 */
@media (max-width: 1180px) {
  .app {
    grid-template-columns: 1fr;
  }

  .panel {
    min-height: auto;
  }
}

@media (max-width: 760px) {
  .app {
    padding: 12px;
  }

  .panel-header {
    padding: 0 18px;
  }

  .panel-title {
    font-size: 23px;
  }
}
</style>
