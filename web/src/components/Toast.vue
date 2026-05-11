<template>
  <transition name="toast">
    <div v-if="show" class="toast">
      <div class="toast-icon">✓</div>
      <div class="toast-content">{{ message }}</div>
    </div>
  </transition>
</template>

<script setup>
defineProps({
  message: {
    type: String,
    default: ''
  },
  show: {
    type: Boolean,
    default: false
  }
})
</script>

<style scoped>
.toast {
  position: fixed;
  right: 24px;
  bottom: 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 16px 24px;
  border-radius: 16px;
  box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
  z-index: 9999;
  max-width: 520px;
  min-width: 300px;
  display: flex;
  align-items: center;
  gap: 12px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.toast-icon {
  width: 32px;
  height: 32px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  font-weight: bold;
  flex-shrink: 0;
}

.toast-content {
  flex: 1;
  line-height: 1.6;
  font-size: 15px;
  font-weight: 500;
}

/* 过渡动画 */
.toast-enter-active,
.toast-leave-active {
  transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

.toast-enter-from {
  opacity: 0;
  transform: translateX(100px) scale(0.8);
}

.toast-leave-to {
  opacity: 0;
  transform: translateX(100px) scale(0.8);
}

.toast-enter-to,
.toast-leave-from {
  opacity: 1;
  transform: translateX(0) scale(1);
}
</style>
