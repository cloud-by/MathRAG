import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { describe, expect, it } from 'vitest'

import MathContent from './MathContent.vue'

async function renderMath(content: string) {
  const wrapper = mount(MathContent, { props: { content } })
  await nextTick()
  await nextTick()
  return wrapper
}

describe('MathContent', () => {
  it('renders inline and display formulas with local KaTeX', async () => {
    const wrapper = await renderMath(
      String.raw`勾股定理 \(a^2+b^2=c^2\)，因此 \[c=\sqrt{a^2+b^2}\]。`,
    )

    expect(wrapper.findAll('.katex')).toHaveLength(2)
    expect(wrapper.find('.katex-display').exists()).toBe(true)
    expect(wrapper.text()).toContain('勾股定理')
  })

  it('leaves escaped currency text intact while rendering dollar math', async () => {
    const wrapper = await renderMath(String.raw`价格是 \$100，变量为 $x+1$。`)

    expect(wrapper.findAll('.katex')).toHaveLength(1)
    expect(wrapper.text()).toContain('$100')
  })

  it('never interprets hostile HTML from answer text', async () => {
    const content = '<script>alert(1)</script><img src=x onerror=alert(2)>'
    const wrapper = await renderMath(content)

    expect(wrapper.find('script').exists()).toBe(false)
    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.text()).toContain('<script>alert(1)</script>')
    expect(wrapper.text()).toContain('<img src=x onerror=alert(2)>')
  })

  it('falls back to visible source text for invalid LaTeX', async () => {
    const wrapper = await renderMath(String.raw`错误公式：\(\left( x\)`)

    expect(wrapper.find('.katex-error').exists()).toBe(true)
    expect(wrapper.text()).toContain(String.raw`\left( x`)
  })

  it('contains long display formulas in the math content scroller', async () => {
    const formula = `\\[${'x+'.repeat(300)}0\\]`
    const wrapper = await renderMath(formula)

    expect(wrapper.find('.math-content__body').classes()).toContain(
      'math-content__body--scrollable',
    )
    expect(
      wrapper.find('.math-content__body').find('.katex-display').exists(),
    ).toBe(true)
  })

  it('renders empty text without synthetic content', async () => {
    const wrapper = await renderMath('')

    expect(wrapper.text()).toBe('')
    expect(wrapper.find('.katex').exists()).toBe(false)
  })
})
