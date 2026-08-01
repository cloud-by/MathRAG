import { expect, test } from '@playwright/test'

import { IDS, installMockApi } from './fixtures'

test('administrator creates a teacher', async ({ page }) => {
  await installMockApi(page, { role: 'admin' })
  await page.goto('/users/new')

  await page.getByLabel('用户名').fill('teacher-a')
  await page.getByLabel('角色').selectOption('teacher')
  await page.getByLabel('临时密码', { exact: true }).fill('temporary-123')
  await page.getByLabel('确认临时密码', { exact: true }).fill('temporary-123')
  await page.getByRole('button', { name: '创建账号' }).click()

  await expect(page).toHaveURL(`/users/${IDS.createdUser}`)
  await expect(page.getByLabel('用户名')).toHaveValue('teacher-a')
  await expect(page.getByLabel('角色')).toHaveValue('teacher')
})

test('teacher sees only owned students', async ({ page }) => {
  await installMockApi(page, { role: 'teacher' })
  await page.goto('/users')

  await expect(page.getByRole('link', { name: 'owned-student' })).toBeVisible()
  await expect(page.getByText('other-student')).toHaveCount(0)
  await expect(page.getByLabel('角色')).toHaveCount(0)
})

test('student cannot open user management', async ({ page }) => {
  await installMockApi(page, { role: 'student' })
  await page.goto('/users')

  await expect(page).toHaveURL('/chat')
  await expect(page.getByRole('link', { name: '用户管理' })).toHaveCount(0)
})

test('temporary password must be changed before work', async ({ page }) => {
  await installMockApi(page, {
    role: 'teacher',
    mustChangePassword: true,
  })
  await page.goto('/chat')

  await expect(page).toHaveURL('/change-password')
  await page.getByLabel('当前密码').fill('temporary-123')
  await page.getByLabel('新密码', { exact: true }).fill('permanent-456')
  await page.getByLabel('确认新密码', { exact: true }).fill('permanent-456')
  await page.getByRole('button', { name: '修改密码' }).click()

  await expect(page).toHaveURL(/\/login\?password_changed=1/)
  await expect(page.getByRole('status')).toContainText('密码已修改')
})
