import { expect, test } from '@playwright/test'

import { IDS, installMockApi } from './fixtures'

test('protects routes, handles login failure/success, and logs out', async ({
  page,
}) => {
  await installMockApi(page, { loggedIn: false })

  await page.goto('/chat')
  await expect(page).toHaveURL(/\/login\?next=\/chat/)

  await page.getByLabel('邮箱或用户名').fill('invalid')
  await page.getByLabel('密码').fill('wrong-password')
  await page.getByRole('button', { name: '登录' }).click()
  await expect(page.getByRole('alert')).toContainText('用户名或密码错误')

  await page.getByLabel('邮箱或用户名').fill('admin')
  await page.getByLabel('密码').fill('correct-password')
  await page.getByRole('button', { name: '登录' }).click()
  await expect(page).toHaveURL(/\/chat$/)
  await expect(
    page.getByRole('heading', { name: '今天想解决什么问题？' }),
  ).toBeVisible()

  await page.getByRole('button', { name: /admin，打开用户菜单/ }).click()
  await page.getByRole('menuitem', { name: '退出登录' }).click()
  await expect(page).toHaveURL(/\/login$/)
})

test('creates a conversation and renders math, reasoning, and references', async ({
  page,
}) => {
  await installMockApi(page)
  await page.goto('/chat')

  await page.getByLabel('数学问题').fill('如何解一元二次方程？')
  await page.getByRole('button', { name: '发送问题' }).click()

  await expect(page).toHaveURL(`/conversations/${IDS.conversation}`)
  await expect(page.getByText(/使用求根公式/).first()).toBeVisible()
  await expect(page.locator('.katex').first()).toBeVisible()
  await page.getByText('解题过程').click()
  await expect(page.getByText('先识别系数，再计算判别式。')).toBeVisible()
  await expect(page.getByText('一元二次方程求根公式').first()).toBeVisible()
})

test('cancels an in-flight answer and retries a transport failure idempotently', async ({
  page,
}) => {
  const state = await installMockApi(page)
  state.holdNextChat = true
  await page.goto('/chat')

  await page.getByLabel('数学问题').fill('请暂停这个回答')
  await page.getByRole('button', { name: '发送问题' }).click()
  await expect(page.getByRole('button', { name: '停止回答' })).toBeVisible()
  await page.getByRole('button', { name: '停止回答' }).click()
  state.releaseChat?.()
  await expect(page.getByText('已停止回答')).toBeVisible()

  state.failNextChat = true
  await page.getByLabel('数学问题').fill('请重试这个回答')
  await page.getByRole('button', { name: '发送问题' }).click()
  await expect(page.getByText('问答服务暂不可用。')).toBeVisible()
  await page.getByRole('button', { name: '使用原请求重试' }).click()
  await expect(page.getByText(/使用求根公式/).first()).toBeVisible()
  expect(state.chatRequestIds.at(-1)).toBe(state.chatRequestIds.at(-2))
})
