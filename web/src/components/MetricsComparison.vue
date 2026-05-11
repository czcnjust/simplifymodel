<template>
  <div class="section">
    <div class="section-title">模型指标</div>

    <!-- 当前选中模型的指标 -->
    <div v-if="currentMetrics" class="current-model-metrics">
      <h3>{{ currentModelName }}</h3>
      <div v-html="metricsHtml(currentMetrics)"></div>
    </div>

    <div v-else class="empty-card">
      当前还没有选择模型。<br />
      请在 Netron 标签页中选择一个模型查看其指标信息。
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  currentMetrics: Object  // 当前选中模型的 metrics
})

// 计算当前模型名称
const currentModelName = computed(() => {
  if (!props.currentMetrics) return '未选择模型'
  const quantization = props.currentMetrics.quantization || ''
  const compression = props.currentMetrics.compression || ''
  
  if (quantization === 'PTQ') return '🔧 PTQ 量化模型'
  if (quantization === 'QAT') return '🎯 QAT 量化感知训练模型'
  if (compression === 'pruning') return '✂️ 剪枝压缩模型'
  
  // 尝试从 run_type 推断模型名称
  return '📊 模型指标'
})

function metricsHtml(m) {
  if (!m) {
    return '<div class="hint">暂无指标</div>'
  }

  let html = `
    <div class="metric-row"><span>准确率</span><span>${formatNumber(m.accuracy, 4)}</span></div>
  `
  
  // 如果是剪枝模型，显示剪枝相关指标
  if (m.compression === 'pruning') {
    html += `
      <div class="metric-row"><span>剪枝前参数量</span><span>${formatInt(m.params_before)}</span></div>
      <div class="metric-row"><span>剪枝后参数量</span><span>${formatInt(m.params_after)}</span></div>
      <div class="metric-row"><span>参数减少比例</span><span>${formatNumber(m.params_reduced_percent, 2)}%</span></div>
      <div class="metric-row"><span>剪枝率</span><span>${formatNumber(m.pruning_ratio * 100, 2)}%</span></div>
      <div class="metric-row"><span>PTH 大小</span><span>${formatNumber(m.model_size_mb, 4)} MB</span></div>
      <div class="metric-row highlight"><span>PyTorch 推理耗时</span><span>${formatNumber(m.inference_time_pytorch_ms || m.inference_time_ms, 2)} ms</span></div>
    `
    
    // 如果有 ONNX Runtime 推理时间，显示出来
    if (m.inference_time_onnx_ms) {
      const speedup = m.onnx_speedup_vs_pytorch || 0
      const speedupIcon = speedup > 1 ? ' ⭐' : ''
      html += `
        <div class="metric-row highlight onnx"><span>ONNX Runtime 推理耗时</span><span>${formatNumber(m.inference_time_onnx_ms, 2)} ms${speedupIcon}</span></div>
      `
      if (speedup > 0) {
        html += `
          <div class="metric-row speedup"><span>ONNX 加速比</span><span>${formatNumber(speedup, 2)}x</span></div>
        `
      }
    }
  }
  // 如果是量化模型（PTQ/QAT），显示量化相关指标
  else if (m.quantization && (m.quantization.includes('PTQ') || m.quantization.includes('QAT'))) {
    html += `
      <div class="metric-row"><span>参数量</span><span>${formatInt(m.params)}</span></div>
      <div class="metric-row"><span>量化方式</span><span>${m.quantization}</span></div>
      <div class="metric-row"><span>训练设备</span><span>${m.device ?? 'CPU'}</span></div>
      <div class="metric-row"><span>微调轮数</span><span>${m.epochs ?? 0}</span></div>
      <div class="metric-row"><span>PTH 大小</span><span>${formatNumber(m.model_size_mb, 4)} MB</span></div>
      <div class="metric-row highlight"><span>PyTorch 推理耗时</span><span>${formatNumber(m.inference_time_pytorch_ms || m.inference_time_ms, 2)} ms</span></div>
    `
    
    // 如果有 ONNX Runtime 推理时间，显示出来
    if (m.inference_time_onnx_ms) {
      const speedup = m.onnx_speedup_vs_pytorch || 0
      const speedupIcon = speedup > 1 ? ' ⭐' : ''
      html += `
        <div class="metric-row highlight onnx"><span>ONNX Runtime 推理耗时</span><span>${formatNumber(m.inference_time_onnx_ms, 2)} ms${speedupIcon}</span></div>
      `
      if (speedup > 0) {
        html += `
          <div class="metric-row speedup"><span>ONNX 加速比</span><span>${formatNumber(speedup, 2)}x</span></div>
        `
      }
    }
  }
  // 普通训练模型
  else {
    html += `
      <div class="metric-row"><span>参数量</span><span>${formatInt(m.params)}</span></div>
      <div class="metric-row"><span>PTH 大小</span><span>${formatNumber(m.model_size_mb, 4)} MB</span></div>
      <div class="metric-row"><span>ONNX 大小</span><span>${formatNumber(m.onnx_size_mb, 4)} MB</span></div>
      <div class="metric-row"><span>训练轮数</span><span>${m.epochs ?? '--'}</span></div>
      <div class="metric-row"><span>训练设备</span><span>${m.device ?? '--'}</span></div>
      <div class="metric-row highlight"><span>PyTorch 推理耗时</span><span>${formatNumber(m.inference_time_pytorch_ms || m.inference_time_ms, 2)} ms</span></div>
    `
    
    // 如果有 ONNX Runtime 推理时间，显示出来
    if (m.inference_time_onnx_ms) {
      const speedup = m.onnx_speedup_vs_pytorch || 0
      const speedupIcon = speedup > 1 ? ' ⭐' : ''
      html += `
        <div class="metric-row highlight onnx"><span>ONNX Runtime 推理耗时</span><span>${formatNumber(m.inference_time_onnx_ms, 2)} ms${speedupIcon}</span></div>
      `
      if (speedup > 0) {
        html += `
          <div class="metric-row speedup"><span>ONNX 加速比</span><span>${formatNumber(speedup, 2)}x</span></div>
        `
      }
    }
  }
  
  return html
}

function formatPercent(v) {
  const sign = v > 0 ? '+' : ''
  return sign + v.toFixed(2) + '%'
}

function formatNumber(v, n) {
  if (v === undefined || v === null || Number.isNaN(Number(v))) {
    return '--'
  }
  return Number(v).toFixed(n)
}

function formatInt(v) {
  if (v === undefined || v === null || Number.isNaN(Number(v))) {
    return '--'
  }
  return Number(v).toLocaleString()
}
</script>

<style scoped>
.section {
  margin-bottom: 24px;
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

.current-model-metrics {
  margin-bottom: 20px;
  padding: 18px;
  border-radius: var(--radius-lg);
  border: 2px solid var(--blue);
  background: var(--blue-soft);
}

.current-model-metrics h3 {
  margin: 0 0 14px;
  font-size: 20px;
  font-weight: 950;
  color: var(--blue);
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

.metric-row {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  padding: 10px 0;
  border-bottom: 1px dashed var(--border);
  font-size: 16px;
}

.metric-row:last-child {
  border-bottom: none;
}

.metric-row span:first-child {
  color: var(--muted);
}

.metric-row span:last-child {
  font-weight: 900;
}

/* 高亮行（PyTorch 和 ONNX） */
.metric-row.highlight {
  background: rgba(59, 130, 246, 0.05);
  padding: 12px 10px;
  border-radius: 8px;
  margin: 4px -10px;
}

/* ONNX Runtime 特别高亮 */
.metric-row.highlight.onnx {
  background: rgba(34, 197, 94, 0.08);
  border-left: 3px solid #22c55e;
}

/* 加速比行 */
.metric-row.speedup {
  background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
  padding: 12px 10px;
  border-radius: 8px;
  margin: 4px -10px;
  border-left: 3px solid #10b981;
}

.metric-row.speedup span:last-child {
  color: #10b981;
  font-size: 18px;
}

.hint {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
}
</style>
