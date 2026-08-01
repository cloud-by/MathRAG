import { expect, test } from '@playwright/test'

import { IDS, installMockApi } from './fixtures'

const viewports = [
  { name: 'mobile', width: 390, height: 844 },
  { name: 'tablet', width: 1024, height: 768 },
  { name: 'desktop', width: 1440, height: 900 },
]

for (const viewport of viewports) {
  test(`${viewport.name} keeps core routes within the document viewport`, async ({
    page,
  }) => {
    await page.setViewportSize(viewport)
    await installMockApi(page)

    for (const path of [
      `/conversations/${IDS.conversation}`,
      '/conversations',
      '/users',
      '/users/new',
      '/knowledge',
      '/documents',
      '/jobs',
    ]) {
      await page.goto(path)
      await expect(page.locator('main').first()).toBeVisible()
      const widths = await page.evaluate(() => ({
        client: document.documentElement.clientWidth,
        scroll: document.documentElement.scrollWidth,
      }))
      expect(widths.scroll, `${path} should not overflow`).toBeLessThanOrEqual(
        widths.client,
      )
    }

    if (viewport.width <= 900) {
      await page.getByRole('button', { name: '打开导航' }).click()
      await expect(page.getByRole('dialog', { name: '主导航' })).toBeVisible()
      await page.getByRole('button', { name: '关闭导航' }).click()
    } else {
      await expect(
        page.getByRole('navigation', { name: '主导航' }),
      ).toBeVisible()
    }

    await page.goto(`/conversations/${IDS.conversation}`)
    const message = page.getByText(/使用求根公式/).first()
    const composer = page.getByLabel('提问输入')
    await message.scrollIntoViewIfNeeded()
    const [messageBox, composerBox] = await Promise.all([
      message.boundingBox(),
      composer.boundingBox(),
    ])
    expect(messageBox).not.toBeNull()
    expect(composerBox).not.toBeNull()
    expect(messageBox!.y + messageBox!.height).toBeLessThanOrEqual(
      composerBox!.y + 1,
    )

    const finalReference = page
      .getByRole('heading', { name: '参考知识' })
      .locator('..')
      .getByRole('listitem')
      .last()
    await finalReference.scrollIntoViewIfNeeded()
    const [referenceBox, visibleComposerBox] = await Promise.all([
      finalReference.boundingBox(),
      composer.boundingBox(),
    ])
    expect(referenceBox).not.toBeNull()
    expect(visibleComposerBox).not.toBeNull()
    expect(referenceBox!.y + referenceBox!.height).toBeLessThanOrEqual(
      visibleComposerBox!.y + 1,
    )
  })
}
