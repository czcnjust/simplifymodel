<template>
  <div class="metric-change-card">
    <div class="metric-change-title">{{ title }}</div>
    <div class="metric-change-value">{{ value }}</div>
    <div class="metric-change-sub">{{ subText }}</div>
  </div>
</template>

<script setup>
defineProps({
  title: {
    type: String,
    required: true
  },
  value: {
    type: String,
    required: true
  },
  subText: {
    type: String,
    required: true
  }
})
</script>

<style scoped>
.metric-change-card {
  border: 1px solid var(--border);
  background: var(--panel-soft);
  border-radius: var(--radius-lg);
  padding: 20px;
}

.metric-change-title {
  color: var(--muted);
  font-size: 15px;
  margin-bottom: 8px;
}

.metric-change-value {
  font-size: 32px;
  font-weight: 950;
  margin-bottom: 8px;
  letter-spacing: -0.8px;
}

.metric-change-sub {
  color: var(--muted);
  font-size: 15px;
}
</style>
