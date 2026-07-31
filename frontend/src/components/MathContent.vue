<script setup lang="ts">
import renderMathInElement from 'katex/contrib/auto-render'
import { nextTick, onMounted, ref, watch } from 'vue'

const props = defineProps<{ content: string }>()
const body = ref<HTMLElement | null>(null)

function escapedDollarToken(content: string): string {
  let token = '\uE000MATHRAG_ESCAPED_DOLLAR\uE001'
  while (content.includes(token)) token += '\uE000'
  return token
}

function restoreEscapedDollars(node: Node, token: string): void {
  if (node.nodeType === Node.TEXT_NODE && node.textContent?.includes(token)) {
    node.textContent = node.textContent.replaceAll(token, '$')
    return
  }
  for (const child of node.childNodes) restoreEscapedDollars(child, token)
}

async function renderContent(): Promise<void> {
  await nextTick()
  const target = body.value
  if (!target) return

  const dollarToken = escapedDollarToken(props.content)
  target.textContent = props.content.replaceAll('\\$', dollarToken)
  if (!props.content) return

  try {
    renderMathInElement(target, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
        { left: '$', right: '$', display: false },
      ],
      maxExpand: 1000,
      maxSize: 20,
      preProcess: (math) => math.replaceAll(dollarToken, '\\$'),
      strict: 'warn',
      throwOnError: false,
      trust: false,
    })
    restoreEscapedDollars(target, dollarToken)
  } catch {
    target.textContent = props.content
  }
}

onMounted(renderContent)
watch(() => props.content, renderContent)
</script>

<template>
  <div class="math-content">
    <div ref="body" class="math-content__body math-content__body--scrollable" />
  </div>
</template>

<style scoped>
.math-content {
  min-width: 0;
  max-width: 100%;
  line-height: 1.75;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}

.math-content__body {
  min-width: 0;
  max-width: 100%;
}

.math-content__body--scrollable :deep(.katex-display) {
  max-width: 100%;
  margin: var(--space-3) 0;
  padding: var(--space-1) 0;
  overflow-x: auto;
  overflow-y: hidden;
}

.math-content :deep(.katex-error) {
  color: inherit;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 0.92em;
}
</style>
