// Run: node frontend/scripts/test-theme.mjs
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'

const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8')
const component = readFileSync(new URL('../src/components/ThemeToggle.vue', import.meta.url), 'utf8')
const bootstrap = html.match(/<script>([\s\S]*?)<\/script>/)[1]
const setup = component.match(/<script setup>([\s\S]*?)<\/script>/)[1].replace(/^import .*$/m, '')
const moduleScript = component.match(/<script>([\s\S]*?)<\/script>/)[1]

function scenario(saved, systemDark, blocked = false) {
  const listeners = {}
  const media = { matches: systemDark, addEventListener: (_, cb) => { listeners.media = cb }, removeEventListener() {} }
  const root = { dataset: {}, style: {} }
  const storage = {
    getItem() { if (blocked) throw Error('Blocked'); return saved },
    setItem(_, value) { if (blocked) throw Error('Blocked'); saved = value },
  }
  const context = vm.createContext({ document: { documentElement: root }, localStorage: storage,
    window: { matchMedia: () => media, addEventListener: (_, cb) => { listeners.storage = cb }, removeEventListener() {} },
    ref: value => ({ value }), onMounted: cb => cb(), onUnmounted() {}, assert })
  vm.runInContext(bootstrap, context)
  const initial = root.dataset.theme
  vm.runInContext(moduleScript + setup, context)
  assert.equal(root.dataset.theme, initial)
  return { root, media, listeners, run: script => vm.runInContext(script, context), setSaved: value => { saved = value } }
}

for (const systemDark of [true, false]) {
  const s = scenario(null, systemDark)
  assert.equal(s.root.dataset.theme, systemDark ? 'dark' : 'light')
  s.media.matches = !systemDark
  s.listeners.media({ type: 'change' })
  assert.equal(s.root.dataset.theme, systemDark ? 'light' : 'dark')
}
for (const saved of ['light', 'dark']) {
  const s = scenario(saved, saved === 'light')
  assert.equal(s.root.dataset.theme, saved)
  s.listeners.media({ type: 'change' })
  assert.equal(s.root.dataset.theme, saved)
  s.run('toggleTheme()')
  assert.equal(s.root.dataset.theme, saved === 'dark' ? 'light' : 'dark')
  s.setSaved('dark')
  s.listeners.storage({ type: 'storage', key: 'mirofish-theme' })
  assert.equal(s.root.dataset.theme, 'dark')
  assert.equal(s.root.style.colorScheme, 'dark')
}
assert.equal(scenario('invalid', true).root.dataset.theme, 'dark')
const blocked = scenario(null, false, true)
blocked.run('toggleTheme(); syncTheme({ type: "change" })')
assert.equal(blocked.root.dataset.theme, 'dark')
console.log('PASS: startup, system preference, manual override, cross-tab sync, blocked storage')
