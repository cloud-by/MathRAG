import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { describe, expect, it } from 'vitest'

import type { components } from '../../api/schema'
import AnswerView from './AnswerView.vue'

type ReferenceItem = components['schemas']['ReferenceItem']

const REFERENCE: ReferenceItem = {
  answer_context: String.raw`判别式为 \(\Delta=b^2-4ac\)。`,
  category: '代数',
  chunk_id: 'chunk-1',
  content: '二次方程根的判别式。',
  difficulty: 'medium',
  example: '',
  index: null,
  keywords: ['二次方程', '判别式'],
  metadata: { source: '教材第一章' },
  rank: 1,
  retrieval_text: '判别式',
  score: 0.9234,
  source_id: 'knowledge-1',
  source_line: null,
  steps: ['计算判别式'],
  title: '二次方程判别式',
}

const ANSWER = {
  answer: String.raw`<script>alert(1)</script> 方程的解为 \(x=2\)。`,
  reasoning_content: '先整理方程，再计算判别式。',
  steps: ['移项并合并同类项。', String.raw`计算 \(\Delta\)。`],
  used_knowledge: ['二次方程判别式'],
  related_questions: ['判别式小于零时怎么办？'],
  references: [REFERENCE],
  agentic_plan: {
    strategy: '按概念检索',
    retrieval_queries: ['二次方程 判别式'],
  },
}

describe('AnswerView', () => {
  it('keeps the main answer visible and detailed reasoning collapsed', async () => {
    const wrapper = mount(AnswerView, { props: { answer: ANSWER } })
    await nextTick()
    await nextTick()

    expect(wrapper.get('[data-testid="answer-main"]').text()).toContain(
      '<script>alert(1)</script>',
    )
    expect(wrapper.find('script').exists()).toBe(false)
    expect(wrapper.find('[data-testid="answer-main"] .katex').exists()).toBe(
      true,
    )
    expect(
      wrapper.get('[data-testid="reasoning-details"]').attributes(),
    ).not.toHaveProperty('open')
    expect(
      wrapper.get('[data-testid="agentic-details"]').attributes(),
    ).not.toHaveProperty('open')
  })

  it('shows reference rank, source, snippet, score, and used knowledge', async () => {
    const wrapper = mount(AnswerView, { props: { answer: ANSWER } })
    await nextTick()
    await nextTick()

    const text = wrapper.text()
    expect(text).toContain('二次方程判别式')
    expect(text).toContain('教材第一章')
    expect(text).toContain('0.923')
    expect(text).toContain('判别式为')
    expect(text).toContain('本次使用的知识')
  })

  it('emits only a fill event for related questions', async () => {
    const wrapper = mount(AnswerView, { props: { answer: ANSWER } })

    await wrapper.get('button[data-testid="related-question"]').trigger('click')

    expect(wrapper.emitted('selectRelated')).toEqual([
      ['判别式小于零时怎么办？'],
    ])
  })

  it('omits the reference section when historical data has no references', () => {
    const wrapper = mount(AnswerView, {
      props: {
        answer: {
          answer: '历史回答',
          steps: [],
          used_knowledge: [],
          related_questions: [],
          references: [],
          agentic_plan: null,
        },
      },
    })

    expect(wrapper.text()).not.toContain('参考知识')
  })
})
