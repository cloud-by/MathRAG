import { expect, test } from '@playwright/test'

import { IDS, installMockApi } from './fixtures'

test('renames, archives, and restores conversation history', async ({
  page,
}) => {
  await installMockApi(page)
  await page.goto('/conversations')

  await expect(page.getByRole('link', { name: '二次方程复习' })).toBeVisible()
  await page.getByRole('button', { name: '重命名“二次方程复习”' }).click()
  await page.getByLabel('会话标题').fill('期末复习')
  await page.getByRole('button', { name: '保存' }).click()
  await expect(page.getByRole('link', { name: '期末复习' })).toBeVisible()

  await page.getByRole('link', { name: '期末复习' }).click()
  await expect(page.getByText('如何解一元二次方程？').first()).toBeVisible()
  await expect(page.locator('.katex').first()).toBeVisible()

  await page.goto('/conversations')
  await page.getByRole('button', { name: '归档“期末复习”' }).click()
  await page.getByRole('button', { name: '确认归档' }).click()
  await expect(page.getByText('还没有会话')).toBeVisible()
  await page.getByRole('button', { name: '已归档' }).click()
  await expect(page.getByRole('link', { name: '期末复习' })).toBeVisible()
})

test('creates knowledge and resolves a revision conflict without losing draft', async ({
  page,
}) => {
  const state = await installMockApi(page)
  await page.goto('/knowledge/new')

  await page.getByLabel('标题').fill('函数单调性')
  await page.getByLabel('类别').fill('calculus')
  await page
    .getByRole('textbox', { name: '知识内容' })
    .fill('导数大于零时函数递增。')
  await page.getByLabel('添加关键词').fill('导数')
  await page.getByLabel('添加关键词').press('Enter')
  await page.locator('#knowledge-step-0').fill('计算导数并判断符号')
  await page.getByRole('button', { name: '创建知识条目' }).click()
  await expect(page).toHaveURL(`/knowledge/${IDS.knowledge}`)

  state.conflictNextKnowledgeUpdate = true
  await page.getByLabel('标题').fill('本地函数草稿')
  await page.getByRole('button', { name: '保存更改' }).click()
  await expect(page.getByText(/服务器版本已更新/)).toBeVisible()
  await expect(page.getByLabel('标题')).toHaveValue('本地函数草稿')
  await page.getByRole('button', { name: '保留草稿后重新应用' }).click()
  await expect(page.getByText('修订版本 3')).toBeVisible()
})

test('uploads a document and applies server-allowed job actions', async ({
  page,
}) => {
  await installMockApi(page)
  await page.goto('/documents')

  await page.getByLabel('选择 PDF 文档').setInputFiles({
    name: 'lesson.pdf',
    mimeType: 'application/pdf',
    buffer: Buffer.from('%PDF-1.4 e2e'),
  })
  await page.getByRole('button', { name: '开始上传' }).click()
  await expect(page.getByText('上传成功')).toBeVisible()
  await page.getByRole('link', { name: '查看 lesson.pdf 的摄取任务' }).click()
  await expect(page).toHaveURL(`/jobs?document_id=${IDS.document}`)

  await page.getByRole('button', { name: `取消任务 ${IDS.pendingJob}` }).click()
  const jobList = page.getByLabel('摄取任务列表')
  await expect(jobList.getByText('已取消')).toBeVisible()
  await page.getByRole('button', { name: `重试任务 ${IDS.failedJob}` }).click()
  await expect(jobList.getByText('处理中')).toBeVisible()
})
