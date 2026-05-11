<template>
  <div class="section result-tabs-container">
    <div class="section-title">实验过程与产物</div>

    <div class="tabs">
      <button
        v-for="tab in tabs"
        :key="tab.id"
        :class="['tab-btn', { active: currentTab === tab.id }]"
        @click="handleTabChange(tab.id)"
      >
        {{ tab.label }}
      </button>
    </div>

    <!-- 训练日志 -->
    <div v-show="currentTab === 'log'" class="tab-panel">
      <div class="log-box">{{ logText || '暂无日志' }}</div>
    </div>

    <!-- Netron 图 -->
    <div v-show="currentTab === 'netron'" class="tab-panel">
      <div class="graph-toolbar">
        <div class="model-selector-group">
          <!-- 模型选择下拉框 -->
          <select 
            v-model="selectedModelKey" 
            @change="handleModelSelectChange"
            class="model-select"
          >
            <option value="" disabled>{{ availableModels.length > 0 ? '请选择模型' : '暂无可用模型' }}</option>
            <option 
              v-for="model in availableModels" 
              :key="model.run_type" 
              :value="model.run_type"
            >
              {{ model.name }}
            </option>
          </select>
        </div>
        
        <!-- 刷新按钮 -->
        <button
          class="btn-refresh-netron"
          @click="refreshNetronModels"
          :disabled="isRefreshingNetron"
          title="刷新模型列表"
        >
          {{ isRefreshingNetron ? '刷新中...' : '↻ 刷新' }}
        </button>
        
        <!-- 格式切换按钮 -->
        <div v-if="currentModel && currentModel.formats && currentModel.formats.length > 1" class="format-switcher">
          <span class="format-label">格式：</span>
          <button
            v-for="fmt in currentModel.formats"
            :key="fmt.format"
            :class="['format-btn', { active: currentFormat === fmt.format }]"
            @click="handleFormatChange(fmt.format)"
          >
            {{ fmt.format.toUpperCase() }}
          </button>
        </div>
        
        <div class="graph-hint">
          Netron 会直接读取训练后生成的模型文件（ONNX、PT 或 PTH）。
        </div>
      </div>

      <div v-if="!netronUrl" class="empty-card">
        当前还没有打开 Netron 图。<br />
        请先训练模型，然后选择模型打开对应的 Netron 结构图。
      </div>

      <iframe
        v-if="netronUrl"
        class="netron-frame"
        :src="netronUrl"
      ></iframe>
    </div>

    <!-- 模型源码 -->
    <div v-show="currentTab === 'structure'" class="tab-panel">
      <div class="graph-toolbar">
        <div class="model-selector-group">
          <!-- 模型选择下拉框 -->
          <select 
            v-model="selectedGeneratedModel" 
            @change="handleGeneratedModelChange"
            class="model-select"
          >
            <option value="" disabled>{{ generatedModels.length > 0 ? '请选择模型' : '暂无可用模型' }}</option>
            <option 
              v-for="model in generatedModels" 
              :key="model.filename" 
              :value="model.filename"
            >
              {{ model.name }}
            </option>
          </select>
        </div>
        
        <div class="save-button-container">
          <button
            v-if="selectedGeneratedModel"
            class="btn-refresh"
            @click="refreshCode"
            :disabled="isRefreshing"
            title="从后端重新加载代码"
          >
            {{ isRefreshing ? '刷新中...' : '↻ 刷新' }}
          </button>
          <button
            v-if="selectedGeneratedModel"
            class="btn-save"
            @click="saveCurrentCode"
            :disabled="isSaving"
          >
            {{ isSaving ? '保存中...' : '保存修改' }}
          </button>
          <button
            v-if="selectedGeneratedModel"
            class="btn-delete"
            @click="deleteCurrentModel"
            :disabled="isDeleting"
            title="删除此模型文件"
          >
            {{ isDeleting ? '删除中...' : '🗑 删除' }}
          </button>
        </div>
      </div>
      
      <div v-if="!selectedGeneratedModel" class="empty-card">
        请从下拉框选择一个模型。<br />

      </div>
      
      <div v-else class="code-editor-container">
        <textarea
          v-model="currentCode"
          class="code-editor"
          spellcheck="false"
          @input="onCodeChange"
        ></textarea>
        <div v-if="hasUnsavedChanges" class="unsaved-indicator">
          ⚠️ 有未保存的修改
        </div>
      </div>
    </div>

    <!-- 生成代码 -->
    <div v-show="currentTab === 'code'" class="tab-panel">
      <div class="code-box">
        {{ improvedCode || '暂无生成代码。训练 improved 模型后，这里会显示 improved_model.py。' }}
      </div>
    </div>

    <!-- 文件下载 -->
    <div v-show="currentTab === 'download'" class="tab-panel">
      <div v-if="!baselineReady && !improvedReady" class="empty-card">
        当前还没有可下载文件。<br />
        训练完成后这里会出现 pth、onnx、metrics、structure 和 improved_model.py。
      </div>

      <div v-if="baselineReady || improvedReady" class="download-grid">
        <div v-if="baselineReady" class="download-card">
          <div class="download-title">Baseline 文件</div>
          <div class="download-links">
            <a href="/api/download/baseline/model.pth" target="_blank">PTH</a>
            <a href="/api/download/baseline/model.onnx" target="_blank">ONNX</a>
            <a href="/api/download/baseline/metrics.json" target="_blank">metrics.json</a>
            <a href="/api/download/baseline/structure.txt" target="_blank">structure.txt</a>
          </div>
        </div>

        <div v-if="improvedReady" class="download-card">
          <div class="download-title">Improved 文件</div>
          <div class="download-links">
            <a href="/api/download/improved/model.pth" target="_blank">PTH</a>
            <a href="/api/download/improved/model.onnx" target="_blank">ONNX</a>
            <a href="/api/download/improved/metrics.json" target="_blank">metrics.json</a>
            <a href="/api/download/improved/structure.txt" target="_blank">structure.txt</a>
            <a href="/api/download/improved/improved_model.py" target="_blank">improved_model.py</a>
          </div>
        </div>
      </div>
    </div>

    <!-- 多轮优化入口 -->
    <div v-show="currentTab === 'strategy'" class="tab-panel">
      <div class="empty-card" style="text-align: left">
        这里用于向展示后续可扩展方向。
      </div>

      <div class="strategy-grid">
        <div class="strategy-card">
          <div class="strategy-card-title">多轮 Agent 优化</div>
          <div class="strategy-card-text">
            读取上一轮 metrics 和结构图，让大模型继续生成下一版网络结构。
          </div>
        </div>

        <div class="strategy-card">
          <div class="strategy-card-title">模型剪枝</div>
          <div class="strategy-card-text">
            删除不重要通道或权重，进一步减少参数量和推理成本。
          </div>
        </div>

        <div class="strategy-card">
          <div class="strategy-card-title">模型量化</div>
          <div class="strategy-card-text">
            将 FP32 权重压缩为 INT8 等格式，降低模型大小并提升部署速度。
          </div>
        </div>

        <div class="strategy-card">
          <div class="strategy-card-title">知识蒸馏</div>
          <div class="strategy-card-text">
            用大模型作为 teacher，训练更小的 student 模型保持效果。
          </div>
        </div>

        <div class="strategy-card">
          <div class="strategy-card-title">用户上传任意模型</div>
          <div class="strategy-card-text">
            后续可以从固定 SimpleCNN 扩展到 ResNet、YOLO、Transformer 等结构。
          </div>
        </div>

        <div class="strategy-card">
          <div class="strategy-card-title">部署建议报告</div>
          <div class="strategy-card-text">
            根据模型大小、参数量、ONNX 文件和指标生成部署建议。
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const props = defineProps({
  currentTab: {
    type: String,
    default: 'log'
  },
  logText: {
    type: String,
    default: ''
  },
  netronUrl: {
    type: String,
    default: ''
  },
  structureText: {
    type: String,
    default: ''
  },
  improvedCode: {
    type: String,
    default: ''
  },
  baselineReady: {
    type: Boolean,
    default: false
  },
  improvedReady: {
    type: Boolean,
    default: false
  }
})

const emit = defineEmits(['tab-change', 'netron-open', 'model-select', 'show-toast'])

const currentModelKey = ref('')
const selectedModelKey = ref('')  // 新增：用于下拉框绑定
const currentFormat = ref('')
const currentModel = ref(null)
const availableModels = ref([])
const structureType = ref('baseline')  // 添加 structureType 响应式变量
const generatedModels = ref([])
const selectedGeneratedModel = ref('')
const currentCode = ref('')
const originalCode = ref('')
const hasUnsavedChanges = ref(false)
const isSaving = ref(false)
const isRefreshing = ref(false)
const isRefreshingNetron = ref(false)  // Netron 刷新状态
const isDeleting = ref(false)  // 删除模型状态

// 标签页配置
const tabs = [
  { id: 'log', label: '训练日志' },
  { id: 'netron', label: 'Netron' },
  { id: 'structure', label: '模型源码' }
]

// 加载可用模型列表
async function loadAvailableModels() {
  try {
    const res = await axios.get('/api/netron/models')
    if (res.data.success) {
      availableModels.value = res.data.models
    }
  } catch (e) {
    console.error('加载模型列表失败:', e)
  }
}

// 刷新 Netron 模型列表（带加载状态）
async function refreshNetronModels() {
  isRefreshingNetron.value = true
  try {
    await loadAvailableModels()
    emit('show-toast', '✓ 模型列表已刷新')
  } catch (e) {
    console.error('刷新失败:', e)
    emit('show-toast', '✗ 刷新失败: ' + e.message)
  } finally {
    isRefreshingNetron.value = false
  }
}

// 加载 generated 文件夹中的模型
async function loadGeneratedModels() {
  console.log('[ResultTabs] 开始加载 generated 模型列表...')
  try {
    const res = await axios.get('/api/generated/models')
    console.log('[ResultTabs] API 响应:', res.data)
    if (res.data.success) {
      const oldModelsCount = generatedModels.value.length
      generatedModels.value = res.data.models
      console.log('[ResultTabs] 加载完成，共', res.data.models.length, '个模型')
      
      // 如果之前没有选中模型，且现在有模型了，自动选中第一个
      if (!selectedGeneratedModel.value && res.data.models.length > 0) {
        selectedGeneratedModel.value = res.data.models[0].filename
        console.log('[ResultTabs] 自动选中第一个模型:', res.data.models[0].filename)
        // 触发模型选择事件
        selectGeneratedModel(res.data.models[0])
      }
      // 如果之前有选中模型，但现在不存在了，清空选择
      else if (selectedGeneratedModel.value && !res.data.models.find(m => m.filename === selectedGeneratedModel.value)) {
        console.log('[ResultTabs] 之前选中的模型已不存在，清空选择')
        selectedGeneratedModel.value = ''
        currentCode.value = ''
        originalCode.value = ''
        hasUnsavedChanges.value = false
      }
    }
  } catch (e) {
    console.error('[ResultTabs] 加载 generated 模型失败:', e)
  }
}

// 选择 generated 模型
function selectGeneratedModel(model) {
  selectedGeneratedModel.value = model.filename
  currentCode.value = model.code
  originalCode.value = model.code
  hasUnsavedChanges.value = false
}

// 代码变化时标记为未保存
function onCodeChange() {
  hasUnsavedChanges.value = currentCode.value !== originalCode.value
}

// 从后端刷新代码
async function refreshCode() {
  if (!selectedGeneratedModel.value) {
    return
  }
  
  // 如枟有未保存的修改，提示用户
  if (hasUnsavedChanges.value) {
    const confirmRefresh = window.confirm('当前有未保存的修改，确定要刷新吗？')
    if (!confirmRefresh) {
      return
    }
  }
  
  isRefreshing.value = true
  try {
    // 重新加载模型列表以获取最新代码
    await loadGeneratedModels()
    
    // 找到当前选中的模型并更新代码
    const model = generatedModels.value.find(m => m.filename === selectedGeneratedModel.value)
    if (model) {
      currentCode.value = model.code
      originalCode.value = model.code
      hasUnsavedChanges.value = false
      emit('show-toast', '✓ 代码已刷新')
    }
  } catch (e) {
    console.error('刷新失败:', e)
    emit('show-toast', '✗ 刷新失败: ' + e.message)
  } finally {
    isRefreshing.value = false
  }
}

// 保存当前代码
async function saveCurrentCode() {
  if (!selectedGeneratedModel.value || !hasUnsavedChanges.value) {
    return
  }
  
  isSaving.value = true
  try {
    const res = await axios.post('/api/generated/save', {
      filename: selectedGeneratedModel.value,
      code: currentCode.value
    })
    
    if (res.data.success) {
      originalCode.value = currentCode.value
      hasUnsavedChanges.value = false
      emit('show-toast', res.data.message || '保存成功！')
      // 重新加载模型列表
      await loadGeneratedModels()
    } else {
      emit('show-toast', '保存失败: ' + (res.data.message || '未知错误'))
    }
  } catch (e) {
    console.error('保存失败:', e)
    emit('show-toast', '保存失败: ' + e.message)
  } finally {
    isSaving.value = false
  }
}

// 删除当前模型
async function deleteCurrentModel() {
  if (!selectedGeneratedModel.value) {
    return
  }
  
  // 确认删除
  const confirmDelete = window.confirm(`确定要删除模型 "${selectedGeneratedModel.value}" 及其所有训练结果（包括 PTQ、QAT、剪枝等）吗？此操作不可恢复！`)
  if (!confirmDelete) {
    return
  }
  
  isDeleting.value = true
  try {
    const res = await axios.delete(`/api/generated/delete/${selectedGeneratedModel.value}`)
    
    if (res.data.success) {
      // 构建详细的删除信息
      let message = res.data.message || '删除成功！'
      if (res.data.deleted_items && res.data.deleted_items.length > 1) {
        message += `\n已删除：${res.data.deleted_items.join(', ')}`
      }
      
      emit('show-toast', message)
      // 清空选择
      selectedGeneratedModel.value = ''
      currentCode.value = ''
      originalCode.value = ''
      hasUnsavedChanges.value = false
      // 重新加载模型列表
      await loadGeneratedModels()
    } else {
      emit('show-toast', '删除失败: ' + (res.data.message || '未知错误'))
    }
  } catch (e) {
    console.error('删除失败:', e)
    emit('show-toast', '删除失败: ' + e.message)
  } finally {
    isDeleting.value = false
  }
}

function handleTabChange(tabId) {
  // 如果切换到模型源码标签，自动刷新模型列表
  if (tabId === 'structure') {
    loadGeneratedModels()
  }
  // 如果切换到 Netron 标签，自动刷新可用模型列表
  if (tabId === 'netron') {
    loadAvailableModels()
  }
  emit('tab-change', tabId)
}

// 选择模型（不立即打开 Netron）
function handleModelSelect(model) {
  currentModelKey.value = model.run_type
  selectedModelKey.value = model.run_type  // 同步更新下拉框
  currentModel.value = model
  // 默认使用第一个格式
  if (model.formats && model.formats.length > 0) {
    currentFormat.value = model.formats[0].format
    // 通知父组件当前选中的模型
    emit('model-select', model.run_type)
    // 自动打开默认格式的 Netron
    openNetronWithFormat(model, currentFormat.value)
  }
}

// 下拉框选择模型
function handleModelSelectChange() {
  if (!selectedModelKey.value) return
  
  const model = availableModels.value.find(m => m.run_type === selectedModelKey.value)
  if (model) {
    handleModelSelect(model)
  }
}

// 下拉框选择 generated 模型
function handleGeneratedModelChange() {
  if (!selectedGeneratedModel.value) return
  
  const model = generatedModels.value.find(m => m.filename === selectedGeneratedModel.value)
  if (model) {
    selectGeneratedModel(model)
  }
}

// 切换格式
function handleFormatChange(format) {
  currentFormat.value = format
  if (currentModel.value) {
    openNetronWithFormat(currentModel.value, format)
  }
}

// 使用指定格式打开 Netron
async function openNetronWithFormat(model, format) {
  // 找到对应格式的文件信息
  const formatInfo = model.formats.find(f => f.format === format)
  if (!formatInfo) {
    console.error('未找到格式:', format)
    return
  }
  
  try {
    // 调用 API 启动 Netron（使用查询参数）
    const params = new URLSearchParams()
    params.append('model_key', formatInfo.key)
    params.append('model_name', `${model.name} (${format.toUpperCase()})`)
    params.append('model_path', formatInfo.path)
    
    const res = await axios.post('/api/netron/start?' + params.toString())
    
    if (res.data.success && res.data.available) {
      // 添加时间戳避免缓存
      const finalUrl = res.data.url + `?t=${Date.now()}`
      emit('netron-open', finalUrl)
    } else {
      console.error('启动 Netron 失败:', res.data.message)
    }
  } catch (e) {
    console.error('启动 Netron 失败:', e)
  }
}

// 组件挂载时加载模型列表
onMounted(() => {
  loadAvailableModels()
  loadGeneratedModels()
})

// 暴露方法供父组件调用
defineExpose({
  loadGeneratedModels
})
</script>

<style scoped>
.section {
  margin-bottom: 24px;
}

.result-tabs-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.section-title {
  font-size: 24px;
  font-weight: 900;
  margin-bottom: 18px;
  display: flex;
  align-items: center;
  gap: 10px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.section-title::before {
  content: '';
  width: 5px;
  height: 28px;
  border-radius: 999px;
  background: var(--blue);
}

.tabs {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  padding-bottom: 18px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}

.tab-btn {
  height: 42px;
  padding: 0 18px;
  border-radius: 999px;
  background: #f1f5f9;
  color: #0f172a;
  font-weight: 900;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
}

.tab-btn:hover {
  background: #e2e8f0;
}

.tab-btn.active {
  background: var(--blue);
  color: white;
}

.tab-panel {
  /* v-show 会控制显示/隐藏，不需要 display 样式 */
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.log-box,
.code-box,
.structure-box {
  background: #0f172a;
  color: #f8fafc;
  border-radius: 20px;
  padding: 20px;
  font-family: Consolas, Monaco, 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
  overflow: auto;
  flex: 1;
  min-height: 400px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

.log-box {
  min-height: 400px;
}

.code-editor-container {
  position: relative;
  width: 100%;
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.code-editor {
  width: 100%;
  flex: 1;
  min-height: 400px;
  background: #0f172a;
  color: #f8fafc;
  border: none;
  border-radius: 20px;
  padding: 20px;
  font-family: Consolas, Monaco, 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.65;
  white-space: pre;
  overflow: auto;
  resize: none;
  outline: none;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
  box-sizing: border-box;
}

.code-editor:focus {
  box-shadow: 0 0 0 2px var(--blue);
}

.unsaved-indicator {
  position: absolute;
  top: 10px;
  right: 20px;
  background: rgba(249, 115, 22, 0.9);
  color: white;
  padding: 8px 16px;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 700;
  z-index: 10;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

.save-button-container {
  display: flex;
  align-items: center;
  gap: 12px;
}

.btn-refresh {
  height: 42px;
  padding: 0 20px;
  border-radius: 999px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-weight: 900;
  font-size: 15px;
  border: none;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  align-items: center;
  gap: 8px;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
  position: relative;
  overflow: hidden;
}

.btn-refresh::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.5s;
}

.btn-refresh:hover:not(:disabled)::before {
  left: 100%;
}

.btn-refresh:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.btn-refresh:active:not(:disabled) {
  transform: translateY(0);
  box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
}

.btn-refresh:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.btn-save {
  height: 42px;
  padding: 0 20px;
  border-radius: 999px;
  background: var(--green);
  color: white;
  font-weight: 900;
  font-size: 15px;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 8px;
}

.btn-save:hover:not(:disabled) {
  background: #15803d;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(22, 163, 74, 0.3);
}

.btn-save:active:not(:disabled) {
  transform: translateY(0);
}

.btn-save:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-delete {
  height: 42px;
  padding: 0 20px;
  border-radius: 999px;
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
  color: white;
  font-weight: 900;
  font-size: 15px;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 8px;
  box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2);
}

.btn-delete:hover:not(:disabled) {
  background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(239, 68, 68, 0.3);
}

.btn-delete:active:not(:disabled) {
  transform: translateY(0);
  box-shadow: 0 2px 8px rgba(239, 68, 68, 0.2);
}

.btn-delete:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-refresh-netron {
  height: 42px;
  padding: 0 20px;
  border-radius: 999px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  font-weight: 900;
  font-size: 15px;
  border: none;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  display: flex;
  align-items: center;
  gap: 8px;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
  position: relative;
  overflow: hidden;
}

.btn-refresh-netron::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.5s;
}

.btn-refresh-netron:hover:not(:disabled)::before {
  left: 100%;
}

.btn-refresh-netron:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.btn-refresh-netron:active:not(:disabled) {
  transform: translateY(0);
  box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
}

.btn-refresh-netron:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.empty-card {
  border: 1.5px dashed var(--border-strong);
  border-radius: var(--radius-lg);
  background: #f8fafc;
  padding: 30px;
  color: var(--muted);
  line-height: 1.8;
  text-align: center;
}

.graph-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}

.model-selector-group {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}

.model-select,
.format-select {
  height: 42px;
  padding: 0 16px;
  border-radius: 12px;
  background: #ffffff;
  border: 2px solid var(--border);
  font-weight: 700;
  font-size: 14px;
  color: #0f172a;
  cursor: pointer;
  transition: all 0.2s;
  outline: none;
  min-width: 180px;
}

.model-select:hover,
.format-select:hover {
  border-color: var(--blue);
}

.model-select:focus,
.format-select:focus {
  border-color: var(--blue);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.model-select option,
.format-select option {
  padding: 8px;
}

.format-switcher {
  display: flex;
  align-items: center;
  gap: 8px;
}

.format-label {
  font-size: 14px;
  color: var(--muted);
  font-weight: 600;
}

.format-btn {
  height: 32px;
  padding: 0 12px;
  border-radius: 8px;
  background: #f1f5f9;
  font-weight: 700;
  font-size: 12px;
  color: #0f172a;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
}

.format-btn:hover {
  background: #e2e8f0;
}

.format-btn.active {
  background: var(--blue);
  color: white;
}

.graph-hint {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
}

.switch-group {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.model-buttons {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.switch-btn.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.switch-btn {
  height: 42px;
  padding: 0 16px;
  border-radius: 999px;
  background: #f1f5f9;
  font-weight: 900;
  color: #0f172a;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
}

.switch-btn:hover {
  background: #e2e8f0;
}

.switch-btn.active {
  background: var(--blue);
  color: white;
}

.netron-frame {
  width: 100%;
  flex: 1;
  min-height: 400px;
  border: 1px solid #dbe5f5;
  border-radius: 22px;
  background: #ffffff;
}

.download-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
}

.download-card {
  border: 1px solid var(--border);
  background: #ffffff;
  border-radius: 18px;
  padding: 18px;
}

.download-title {
  font-weight: 950;
  font-size: 18px;
  margin-bottom: 12px;
}

.download-links {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.download-links a {
  text-decoration: none;
  color: white;
  background: var(--dark);
  padding: 10px 13px;
  border-radius: 12px;
  font-weight: 800;
  font-size: 14px;
}

.strategy-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-top: 14px;
}

.strategy-card {
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px;
  background: #ffffff;
}

.strategy-card-title {
  font-weight: 950;
  margin-bottom: 8px;
}

.strategy-card-text {
  color: var(--muted);
  line-height: 1.7;
  font-size: 14px;
}
</style>
