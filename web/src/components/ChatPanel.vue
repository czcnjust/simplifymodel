<template>
  <section class="panel">
    <div class="panel-header">
      <div class="panel-title">基于大模型的边缘智能模型轻量化方法实践平台</div>
      <div class="badge">南京理工大学</div>
    </div>

    <div id="chatBody" class="chat-body">
      <div
        v-for="(message, index) in messages"
        :key="index"
        :class="['message', message.role]"
      >
        <div class="bubble">
          <div v-html="message.text"></div>
          <span v-if="message.streaming" class="streaming-cursor">|</span>
        </div>
      </div>
    </div>

    <div class="composer">
      <div class="composer-label">输入给 Agent 的消息 / 模型代码 / 优化需求：</div>
      
      <!-- 可复制的示例提示 -->
      <div class="example-hints" title="点击即可复制">
        <div class="hint-item" @click="copyHint($event)">将上传的模型修改为包含残差结构，但是要适合量化</div>
        <div class="hint-item" @click="copyHint($event)">将当前模型修改为包含注意力机制，但是要适合量化</div>
        <div class="hint-item" @click="copyHint($event)">训练sample.py，训练轮次3轮</div>
        <div class="hint-item" @click="copyHint($event)">对sample.py进行ptq量化</div>
        <div class="hint-item" @click="copyHint($event)">对sample.py进行qat量化，训练2轮</div>
        <div class="hint-item" @click="copyHint($event)">对sample.py进行剪枝，训练2轮，剪枝率40%</div>

      </div>

      <div class="input-wrap">
        <textarea
          id="agentInput"
          v-model="agentInput"
          placeholder="在此输入消息...（可点击上方示例快速填充）"
          @keydown="handleKeydown"
        ></textarea>

        <div class="composer-tools">
          <div class="tool-left">
            <input
              id="fileInput"
              type="file"
              accept=".py"
              style="display:none"
              @change="handleFileChange"
            />
            <button class="btn btn-gray" @click="choosePyFile">上传 .py</button>
            <span id="fileNameHint" class="hint">{{ fileNameHint }}</span>
          </div>

          <div class="tool-right">
            <button
              id="sendBtn"
              class="btn btn-primary"
              @click="handleSend"
              :disabled="isSending"
            >
              发送
            </button>
            <button class="btn btn-green" @click="$emit('clear-chat')">清空对话</button>
          </div>
        </div>
      </div>

      <div :class="['status-box', statusType]">{{ statusText }}</div>

      <div class="flow-text">
        Demo 流程：输入模型代码或上传 .py → 训练 baseline → 输入优化需求 → 大模型生成 improved_model.py → 自动训练 → 展示指标、结构图、代码和模型文件。
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  messages: {
    type: Array,
    required: true
  },
  statusType: {
    type: String,
    default: ''
  },
  statusText: {
    type: String,
    default: '当前状态：空闲'
  },
  isSending: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['send', 'clear-chat', 'file-upload'])

const agentInput = ref('')
const fileNameHint = ref('未选择文件')

function choosePyFile() {
  document.getElementById('fileInput').click()
}

async function handleFileChange(event) {
  const file = event.target.files[0]
  if (!file) {
    fileNameHint.value = '未选择文件'
    return
  }

  fileNameHint.value = file.name

  try {
    const text = await file.text()
    agentInput.value = text
    emit('file-upload', file, text)
  } catch (e) {
    console.error('读取文件失败：', e)
  }
}

function handleSend() {
  const text = agentInput.value.trim()
  if (!text) {
    return
  }
  
  emit('send', text)
  agentInput.value = ''
}

// 处理键盘事件
function handleKeydown(event) {
  console.log('Key pressed:', event.key, 'Ctrl:', event.ctrlKey, 'Meta:', event.metaKey)
  // Ctrl+Enter 或 Cmd+Enter (Mac) 发送消息
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
    console.log('Ctrl+Enter detected, sending message...')
    event.preventDefault()
    handleSend()
  }
}

// 复制提示文本到输入框
function copyHint(event) {
  const text = event.target.textContent
  agentInput.value = text
  // 聚焦到输入框
  document.getElementById('agentInput').focus()
}

// 暴露方法供父组件调用
defineExpose({
  agentInput,
  fileNameHint
})
</script>

<style scoped>
.panel {
  background: var(--panel);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow);
  overflow: hidden;
  height: 100%;
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

.chat-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 24px 28px;
  background: radial-gradient(
      circle at top left,
      rgba(37, 99, 235, 0.06),
      transparent 36%
    ),
    linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}

.message {
  display: flex;
  margin-bottom: 18px;
}

.message.user {
  justify-content: flex-end;
}

.bubble {
  max-width: 86%;
  padding: 16px 18px;
  border-radius: 18px;
  font-size: 16px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.bubble :deep(h1),
.bubble :deep(h2),
.bubble :deep(h3),
.bubble :deep(h4),
.bubble :deep(h5),
.bubble :deep(h6) {
  margin: 12px 0 8px;
  font-weight: 700;
  line-height: 1.4;
}

.bubble :deep(h1) {
  font-size: 1.5em;
}

.bubble :deep(h2) {
  font-size: 1.3em;
}

.bubble :deep(h3) {
  font-size: 1.15em;
}

.bubble :deep(p) {
  margin: 8px 0;
}

.bubble :deep(ul),
.bubble :deep(ol) {
  margin: 8px 0;
  padding-left: 24px;
}

.bubble :deep(li) {
  margin: 4px 0;
}

.bubble :deep(code) {
  background: rgba(0, 0, 0, 0.06);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Courier New', monospace;
  font-size: 0.9em;
}

.bubble :deep(pre) {
  background: #f5f5f5;
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 10px 0;
}

.bubble :deep(pre code) {
  background: transparent;
  padding: 0;
}

.bubble :deep(a) {
  color: var(--blue);
  text-decoration: underline;
}

.bubble :deep(strong) {
  font-weight: 700;
}

.bubble :deep(em) {
  font-style: italic;
}

.bubble :deep(blockquote) {
  border-left: 4px solid var(--border);
  padding-left: 12px;
  margin: 10px 0;
  color: var(--muted);
}

.message.agent .bubble {
  background: #ffffff;
  border: 1px solid var(--border);
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
}

.message.user .bubble {
  background: var(--blue);
  color: #ffffff;
  border: 1px solid var(--blue);
  box-shadow: 0 10px 24px rgba(37, 99, 235, 0.2);
}

.streaming-cursor {
  display: inline-block;
  animation: blink 1s infinite;
  color: var(--blue);
  font-weight: bold;
  margin-left: 2px;
}

@keyframes blink {
  0%, 50% {
    opacity: 1;
  }
  51%, 100% {
    opacity: 0;
  }
}

.composer {
  border-top: 1px solid var(--border);
  background: #ffffff;
  padding: 20px 28px 24px;
  flex-shrink: 0;
}

.composer-label {
  font-size: 17px;
  font-weight: 900;
  margin-bottom: 10px;
}

/* 可复制的示例提示 */
.example-hints {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}

.hint-item {
  padding: 6px 12px;
  background: #f1f5f9;
  border-radius: 8px;
  font-size: 13px;
  color: #475569;
  cursor: pointer;
  transition: all 0.2s;
  user-select: none;
}

.hint-item:hover {
  background: var(--blue-soft);
  color: var(--blue);
  transform: translateY(-1px);
  box-shadow: 0 2px 8px rgba(37, 99, 235, 0.15);
}

.hint-item:active {
  transform: translateY(0);
}

.input-wrap {
  border: 1.5px solid var(--border-strong);
  border-radius: 20px;
  background: #ffffff;
  overflow: hidden;
  transition: 0.2s;
}

.input-wrap:focus-within {
  border-color: var(--blue);
  box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.1);
}

#agentInput {
  width: 100%;
  height: 190px;
  resize: vertical;
  border: none;
  outline: none;
  padding: 18px 18px 10px;
  font-size: 17px;
  line-height: 1.8;
  background: transparent;
}

.composer-tools {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px 12px;
}

.tool-left,
.tool-right {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
}

.hint {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
}

.btn {
  height: 48px;
  padding: 0 20px;
  border-radius: 14px;
  font-weight: 900;
  font-size: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  white-space: nowrap;
  border: none;
  cursor: pointer;
  transition: opacity 0.2s;
}

.btn:hover:not(:disabled) {
  opacity: 0.9;
}

.btn:active:not(:disabled) {
  opacity: 0.8;
}

.btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.btn-primary {
  background: var(--blue);
  color: white;
}

.btn-green {
  background: var(--green);
  color: white;
}

.btn-gray {
  background: #f1f5f9;
  color: #0f172a;
}

.status-box {
  margin-top: 14px;
  padding: 13px 16px;
  border-radius: 14px;
  font-size: 15px;
  color: #475569;
  background: #f1f5f9;
}

.status-box.running {
  background: var(--blue-soft);
  color: var(--blue);
}

.status-box.success {
  background: var(--green-soft);
  color: #047857;
}

.status-box.failed {
  background: var(--red-soft);
  color: var(--red);
}

.flow-text {
  margin-top: 14px;
  color: var(--muted);
  line-height: 1.7;
  font-size: 15px;
}
</style>
