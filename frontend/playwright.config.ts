import { defineConfig } from '@playwright/test'

const localChrome =
  process.platform === 'win32'
    ? 'C:/Program Files/Google/Chrome/Application/chrome.exe'
    : undefined

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: 'http://127.0.0.1:4173',
    browserName: 'chromium',
    launchOptions:
      !process.env.CI && localChrome ? { executablePath: localChrome } : {},
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
})
