<template>
  <button
    class="theme-toggle"
    type="button"
    role="switch"
    aria-label="Dark mode"
    :aria-checked="dark"
    :title="dark ? 'Switch to light mode' : 'Switch to dark mode'"
    @click="toggleTheme"
  >
    <svg v-if="dark" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" aria-hidden="true">
      <path d="M20.9 13a9 9 0 0 1-9.9-9.9A9 9 0 1 0 20.9 13Z" />
    </svg>
    <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" aria-hidden="true">
      <circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" />
    </svg>
    <span>{{ dark ? 'Dark' : 'Light' }}</span>
    <span class="theme-toggle-track" aria-hidden="true"><span /></span>
  </button>
</template>

<script>
let memoryTheme = null
</script>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const dark = ref(document.documentElement.dataset.theme === 'dark')
const systemTheme = window.matchMedia('(prefers-color-scheme: dark)')

function savedTheme() {
  try { return localStorage.getItem('mirofish-theme') } catch { return memoryTheme }
}

function applyTheme(theme) {
  dark.value = theme === 'dark'
  document.documentElement.dataset.theme = theme
  document.documentElement.style.colorScheme = theme
}

function toggleTheme() {
  const theme = dark.value ? 'light' : 'dark'
  memoryTheme = theme
  applyTheme(theme)
  try { localStorage.setItem('mirofish-theme', theme) } catch { /* Storage may be blocked. */ }
}

function syncTheme(event) {
  if (event.type === 'storage' && event.key !== null && event.key !== 'mirofish-theme') return
  const saved = savedTheme()
  applyTheme(saved === 'dark' || saved === 'light' ? saved : systemTheme.matches ? 'dark' : 'light')
}

onMounted(() => {
  // The system theme may change while a route is loading.
  syncTheme({ type: 'change' })
  systemTheme.addEventListener('change', syncTheme)
  window.addEventListener('storage', syncTheme)
})
onUnmounted(() => {
  systemTheme.removeEventListener('change', syncTheme)
  window.removeEventListener('storage', syncTheme)
})
</script>

<style scoped>
.theme-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-height: 36px;
  padding: 7px 10px;
  border: 1px solid var(--border-strong);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  flex-shrink: 0;
}
.theme-toggle:hover { background: var(--surface-hover); }
.theme-toggle-track { width: 26px; height: 15px; padding: 2px; border-radius: 12px; background: var(--border-strong); }
.theme-toggle-track span { display: block; width: 11px; height: 11px; border-radius: 50%; background: var(--surface); transition: transform 150ms; }
[aria-checked="true"] .theme-toggle-track { background: var(--accent); }
[aria-checked="true"] .theme-toggle-track span { transform: translateX(11px); }
@media (max-width: 480px) { .theme-toggle { gap: 6px; } .theme-toggle-track { display: none; } }
@media (prefers-reduced-motion: reduce) { .theme-toggle-track span { transition: none; } }
</style>
