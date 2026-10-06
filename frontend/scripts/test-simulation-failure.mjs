// Run: node frontend/scripts/test-simulation-failure.mjs
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import vm from 'node:vm'
import * as Vue from 'vue'
import { compile } from '@vue/compiler-dom'
import { parse } from '@vue/compiler-sfc'
import { renderToString } from '@vue/server-renderer'

const source = readFileSync(new URL('../src/components/Step3Simulation.vue', import.meta.url), 'utf8')
const { descriptor } = parse(source)
const setup = descriptor.scriptSetup.content
  .replace(/^import[\s\S]*?from ['"].*?['"]\s*$/gm, '')
const template = descriptor.template.content
const render = new Function('Vue', compile(template, { mode: 'function', prefixIdentifiers: true }).code)(Vue)
const events = [], timers = new Set()
let response
const context = vm.createContext({ ...Vue, console,
  defineProps: () => ({ simulationId: 'sim_test', maxRounds: 1, minutesPerRound: 30, systemLogs: [] }),
  defineEmits: () => (...event) => events.push(event),
  useRouter: () => ({ push() {} }), onMounted() {}, onUnmounted() {}, watch() {},
  startSimulation: async () => ({ success: true, data: { runner_status: 'running' } }),
  getRunStatus: async () => ({ success: true, data: response }),
  setInterval: cb => { timers.add(cb); return cb }, clearInterval: cb => timers.delete(cb),
})
vm.runInContext(setup + '\nglobalThis.state = { props, startError, phase, runStatus, allActions, isStarting, isGeneratingReport, doStartSimulation, handleNextStep, chronologicalActions, twitterElapsedTime, redditElapsedTime };', context)
const state = context.state
const html = () => renderToString(Vue.createSSRApp({ render, setup: () => ({ ...state, ...state.props }) }))

await state.doStartSimulation()
assert.equal(timers.size, 2)
response = { runner_status: 'running' }
await vm.runInContext('fetchRunStatus()', context)
assert.equal(state.phase.value, 1)
assert.match(await html(), /Waiting for agent actions/)

response = { runner_status: 'failed', error: "cannot import name 'FastMCP' from 'mcp.server'" }
await vm.runInContext('fetchRunStatus()', context)
assert.equal(timers.size, 0)
assert.equal(state.phase.value, -1)
assert.equal(events.at(-1)[1], 'error')
assert.ok(events.some(event => event[0] === 'add-log' && event[1].includes(response.error)))
const failed = await html()
assert.match(failed, /role="alert"/)
assert.match(failed, /FastMCP/)
assert.doesNotMatch(failed, /Waiting for agent actions/)
assert.match(failed, /<button[^>]*disabled[^>]*>(?:<!--.*?-->|\s)*Generate Report/)

await state.doStartSimulation()
assert.equal(state.startError.value, null)
assert.equal(timers.size, 2)
response = { runner_status: 'failed' }
await vm.runInContext('fetchRunStatus()', context)
assert.ok(state.startError.value)
assert.equal(timers.size, 0)

await state.doStartSimulation()
response = { runner_status: 'completed' }
await vm.runInContext('fetchRunStatus()', context)
assert.equal(state.phase.value, 2)
assert.equal(events.at(-1)[1], 'completed')
assert.equal(timers.size, 0)
console.log('PASS: failure alert, error status, stopped polling, disabled report, retry and completion')
