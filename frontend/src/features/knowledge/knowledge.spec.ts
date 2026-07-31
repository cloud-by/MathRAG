import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/vue'
import { defineComponent, ref } from 'vue'
import { createMemoryHistory, createRouter, RouterView } from 'vue-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ApiError } from '../../api/errors'
import { knowledgeApi } from './api'
import KeywordInput from './KeywordInput.vue'
import KnowledgeEditorPage from './KnowledgeEditorPage.vue'
import KnowledgeListPage from './KnowledgeListPage.vue'
import StepEditor from './StepEditor.vue'
import type { KnowledgeItem, KnowledgePage } from './types'

const ITEM_ID = '11111111-1111-4111-8111-111111111111'

const ITEM: KnowledgeItem = {
  id: ITEM_ID,
  legacy_id: 'k0001',
  owner_id: '22222222-2222-4222-8222-222222222222',
  category: 'algebra',
  title: '一元二次方程',
  keywords: ['方程', '求根'],
  content: '使用求根公式 $x=\\frac{-b\\pm\\sqrt{b^2-4ac}}{2a}$。',
  example: '$x^2-1=0$',
  steps: ['整理方程', '代入公式'],
  difficulty: 'medium',
  visibility: 'public',
  status: 'ready',
  revision: 7,
  created_at: '2026-07-30T08:00:00Z',
  updated_at: '2026-07-31T10:30:00Z',
}

function page(
  items: KnowledgeItem[] = [ITEM],
  overrides: Partial<KnowledgePage> = {},
): KnowledgePage {
  return {
    items,
    page: 1,
    page_size: 20,
    total: items.length,
    ...overrides,
  }
}

function apiError(
  status: number,
  code: string,
  message: string,
  details: unknown = null,
): ApiError {
  return new ApiError({
    status,
    code,
    message,
    details,
    requestId: `request-${status}`,
  })
}

async function renderPage(path: string) {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/knowledge', name: 'knowledge', component: KnowledgeListPage },
      {
        path: '/knowledge/new',
        name: 'knowledge-new',
        component: KnowledgeEditorPage,
      },
      {
        path: '/knowledge/:id',
        name: 'knowledge-detail',
        component: KnowledgeEditorPage,
      },
    ],
  })
  await router.push(path)
  await router.isReady()
  const Host = defineComponent({
    components: { RouterView },
    template: '<RouterView />',
  })
  render(Host, { global: { plugins: [router] } })
  return router
}

beforeEach(() => {
  vi.spyOn(knowledgeApi, 'list').mockResolvedValue(page())
  vi.spyOn(knowledgeApi, 'get').mockResolvedValue(ITEM)
  vi.spyOn(knowledgeApi, 'create').mockResolvedValue(ITEM)
  vi.spyOn(knowledgeApi, 'update').mockResolvedValue({ ...ITEM, revision: 8 })
  vi.spyOn(knowledgeApi, 'archive').mockResolvedValue(undefined)
})

afterEach(() => vi.restoreAllMocks())

describe('knowledge collection', () => {
  it('loads the real server filters from URL parameters', async () => {
    vi.mocked(knowledgeApi.list).mockResolvedValue(
      page([ITEM], { page: 2, total: 24 }),
    )
    await renderPage(
      '/knowledge?status=ready&visibility=public&category=algebra&page=2',
    )

    expect(
      await screen.findByRole('link', { name: '一元二次方程' }),
    ).toBeTruthy()
    expect(knowledgeApi.list).toHaveBeenCalledWith(
      {
        status: 'ready',
        visibility: 'public',
        category: 'algebra',
        page: 2,
        pageSize: 20,
      },
      expect.any(AbortSignal),
    )
  })

  it('archives with the current revision only after confirmation', async () => {
    await renderPage('/knowledge')
    await screen.findByText('一元二次方程')

    await fireEvent.click(
      screen.getByRole('button', { name: '归档“一元二次方程”' }),
    )
    const dialog = screen.getByRole('alertdialog', { name: '归档知识条目' })
    expect(knowledgeApi.archive).not.toHaveBeenCalled()
    await fireEvent.click(
      within(dialog).getByRole('button', { name: '确认归档' }),
    )

    await waitFor(() =>
      expect(knowledgeApi.archive).toHaveBeenCalledWith(ITEM_ID, 7),
    )
    await waitFor(() => expect(knowledgeApi.list).toHaveBeenCalledTimes(2))
  })

  it('surfaces archive conflicts without removing the server row', async () => {
    vi.mocked(knowledgeApi.archive).mockRejectedValueOnce(
      apiError(
        409,
        'KNOWLEDGE_REVISION_CONFLICT',
        '知识条目已被其他操作更新。',
      ),
    )
    await renderPage('/knowledge')
    await screen.findByText('一元二次方程')
    await fireEvent.click(
      screen.getByRole('button', { name: '归档“一元二次方程”' }),
    )
    await fireEvent.click(screen.getByRole('button', { name: '确认归档' }))

    expect((await screen.findByRole('alert')).textContent).toContain(
      '知识条目已被其他操作更新。',
    )
    expect(screen.getByRole('link', { name: '一元二次方程' })).toBeTruthy()
  })
})

describe('knowledge field editors', () => {
  it('trims, removes empty values, and deduplicates keywords', async () => {
    const Host = defineComponent({
      components: { KeywordInput },
      setup() {
        const keywords = ref(['方程'])
        return { keywords }
      },
      template: '<KeywordInput v-model="keywords" />',
    })
    render(Host)
    const input = screen.getByRole('textbox', { name: '添加关键词' })

    await fireEvent.update(input, '  判别式  ')
    await fireEvent.keyDown(input, { key: 'Enter' })
    await fireEvent.update(input, '判别式')
    await fireEvent.keyDown(input, { key: 'Enter' })
    await fireEvent.update(input, '   ')
    await fireEvent.keyDown(input, { key: 'Enter' })

    expect(
      screen.getAllByTestId('keyword-token').map((node) => node.textContent),
    ).toEqual(['方程', '判别式'])
  })

  it('adds, removes, and reorders steps as an ordered string array', async () => {
    const Host = defineComponent({
      components: { StepEditor },
      setup() {
        const steps = ref(['第一步', '第二步'])
        return { steps }
      },
      template: '<StepEditor v-model="steps" />',
    })
    render(Host)

    await fireEvent.click(screen.getByRole('button', { name: '上移步骤 2' }))
    expect(
      (screen.getByRole('textbox', { name: '步骤 1' }) as HTMLInputElement)
        .value,
    ).toBe('第二步')
    await fireEvent.click(screen.getByRole('button', { name: '添加步骤' }))
    expect(screen.getByRole('textbox', { name: '步骤 3' })).toBeTruthy()
    await fireEvent.click(screen.getByRole('button', { name: '删除步骤 2' }))
    expect(screen.queryByRole('textbox', { name: '步骤 3' })).toBeNull()
  })
})

describe('knowledge editor', () => {
  it('creates with exact OpenAPI fields and opens the server entity', async () => {
    const router = await renderPage('/knowledge/new')
    await fireEvent.update(
      screen.getByRole('textbox', { name: '标题' }),
      '函数单调性',
    )
    await fireEvent.update(
      screen.getByRole('textbox', { name: '类别' }),
      'calculus',
    )
    await fireEvent.update(
      screen.getByRole('textbox', { name: '知识内容' }),
      '导数大于零时函数递增。',
    )
    const keyword = screen.getByRole('textbox', { name: '添加关键词' })
    await fireEvent.update(keyword, '导数')
    await fireEvent.keyDown(keyword, { key: 'Enter' })
    await fireEvent.update(
      screen.getByRole('textbox', { name: '步骤 1' }),
      '求导',
    )
    await fireEvent.click(screen.getByRole('button', { name: '创建知识条目' }))

    expect(knowledgeApi.create).toHaveBeenCalledWith({
      category: 'calculus',
      title: '函数单调性',
      keywords: ['导数'],
      content: '导数大于零时函数递增。',
      example: '',
      steps: ['求导'],
      difficulty: 'medium',
      visibility: 'public',
    })
    await waitFor(() =>
      expect(router.currentRoute.value.path).toBe(`/knowledge/${ITEM_ID}`),
    )
  })

  it('updates with the baseline revision and adopts the server revision', async () => {
    await renderPage(`/knowledge/${ITEM_ID}`)
    const title = await screen.findByRole('textbox', { name: '标题' })
    await fireEvent.update(title, '一元二次方程新解法')
    await fireEvent.click(screen.getByRole('button', { name: '保存更改' }))

    expect(knowledgeApi.update).toHaveBeenCalledWith(
      ITEM_ID,
      expect.objectContaining({
        revision: 7,
        title: '一元二次方程新解法',
      }),
    )
    expect(await screen.findByText('修订版本 8')).toBeTruthy()
  })

  it('maps server 422 details to fields', async () => {
    vi.mocked(knowledgeApi.update).mockRejectedValueOnce(
      apiError(422, 'REQUEST_VALIDATION_FAILED', '请求参数校验失败。', [
        { loc: ['body', 'category'], msg: '类别格式不正确' },
      ]),
    )
    await renderPage(`/knowledge/${ITEM_ID}`)
    const category = await screen.findByRole('textbox', { name: '类别' })
    await fireEvent.update(category, 'geometry')
    await fireEvent.click(screen.getByRole('button', { name: '保存更改' }))

    expect(await screen.findByText('类别格式不正确')).toBeTruthy()
    expect(category.getAttribute('aria-invalid')).toBe('true')
  })

  it('preserves the local draft and reapplies it over an explicit server revision', async () => {
    const serverVersion = {
      ...ITEM,
      title: '服务器标题',
      content: '服务器内容',
      revision: 8,
    }
    vi.mocked(knowledgeApi.get)
      .mockResolvedValueOnce(ITEM)
      .mockResolvedValueOnce(serverVersion)
    vi.mocked(knowledgeApi.update)
      .mockRejectedValueOnce(
        apiError(
          409,
          'KNOWLEDGE_REVISION_CONFLICT',
          '知识条目已被其他操作更新。',
        ),
      )
      .mockResolvedValueOnce({
        ...serverVersion,
        title: '本地草稿',
        revision: 9,
      })

    await renderPage(`/knowledge/${ITEM_ID}`)
    const title = await screen.findByRole('textbox', { name: '标题' })
    await fireEvent.update(title, '本地草稿')
    await fireEvent.click(screen.getByRole('button', { name: '保存更改' }))

    expect((await screen.findByRole('alert')).textContent).toContain(
      '服务器版本已更新',
    )
    expect((title as HTMLInputElement).value).toBe('本地草稿')
    expect(screen.getByText('服务器版本：修订 8')).toBeTruthy()

    await fireEvent.click(
      screen.getByRole('button', { name: '保留草稿后重新应用' }),
    )
    expect(knowledgeApi.update).toHaveBeenLastCalledWith(
      ITEM_ID,
      expect.objectContaining({ revision: 8, title: '本地草稿' }),
    )
    expect(await screen.findByText('修订版本 9')).toBeTruthy()
  })

  it('can discard the draft and reload the conflicting server version', async () => {
    const serverVersion = { ...ITEM, title: '服务器标题', revision: 8 }
    vi.mocked(knowledgeApi.get)
      .mockResolvedValueOnce(ITEM)
      .mockResolvedValueOnce(serverVersion)
    vi.mocked(knowledgeApi.update).mockRejectedValueOnce(
      apiError(
        409,
        'KNOWLEDGE_REVISION_CONFLICT',
        '知识条目已被其他操作更新。',
      ),
    )

    await renderPage(`/knowledge/${ITEM_ID}`)
    const title = await screen.findByRole('textbox', { name: '标题' })
    await fireEvent.update(title, '将被放弃的草稿')
    await fireEvent.click(screen.getByRole('button', { name: '保存更改' }))
    await screen.findByText('服务器版本：修订 8')
    await fireEvent.click(
      screen.getByRole('button', { name: '重新载入服务器版本' }),
    )

    expect((title as HTMLInputElement).value).toBe('服务器标题')
    expect(screen.queryByText('服务器版本：修订 8')).toBeNull()
  })

  it('shows permission errors without exposing an editor', async () => {
    vi.mocked(knowledgeApi.get).mockRejectedValueOnce(
      apiError(403, 'AUTH_FORBIDDEN', '当前账户没有此操作权限。'),
    )
    await renderPage(`/knowledge/${ITEM_ID}`)

    expect((await screen.findByRole('alert')).textContent).toContain(
      '当前账户没有此操作权限。',
    )
    expect(screen.queryByRole('textbox', { name: '标题' })).toBeNull()
  })

  it('blocks route changes while the form is dirty and the user declines', async () => {
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
    const router = await renderPage(`/knowledge/${ITEM_ID}`)
    const title = await screen.findByRole('textbox', { name: '标题' })
    await fireEvent.update(title, '尚未保存的标题')

    await router.push('/knowledge')

    expect(confirm).toHaveBeenCalledTimes(1)
    expect(router.currentRoute.value.path).toBe(`/knowledge/${ITEM_ID}`)
  })
})
