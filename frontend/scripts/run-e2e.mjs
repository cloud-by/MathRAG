import { spawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const frontendRoot = fileURLToPath(new URL('../', import.meta.url))
const serverUrl = 'http://127.0.0.1:4173/login'
const viteCli = path.join(
  frontendRoot,
  'node_modules',
  'vite',
  'bin',
  'vite.js',
)
const playwrightCli = path.join(
  frontendRoot,
  'node_modules',
  '@playwright',
  'test',
  'cli.js',
)

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds))
}

async function serverIsReady() {
  try {
    const response = await fetch(serverUrl, {
      signal: AbortSignal.timeout(1_000),
    })
    return response.ok
  } catch {
    return false
  }
}

async function waitForServer(server) {
  const deadline = Date.now() + 120_000
  while (Date.now() < deadline) {
    if (await serverIsReady()) return
    if (server.exitCode !== null) {
      throw new Error(`Vite exited before becoming ready (${server.exitCode}).`)
    }
    await delay(200)
  }
  throw new Error(`Vite did not become ready at ${serverUrl}.`)
}

function waitForExit(child) {
  if (child.exitCode !== null) return Promise.resolve(child.exitCode)
  return new Promise((resolve, reject) => {
    child.once('error', reject)
    child.once('exit', (code) => resolve(code ?? 1))
  })
}

async function stopServer(server) {
  if (server.exitCode !== null) return
  server.kill()
  await Promise.race([waitForExit(server), delay(5_000)])
}

let server = null

try {
  const existingServerIsReady = await serverIsReady()
  if (existingServerIsReady && process.env.CI) {
    throw new Error(`Port 4173 is already serving ${serverUrl} in CI.`)
  }

  if (!existingServerIsReady) {
    server = spawn(
      process.execPath,
      [viteCli, '--host', '127.0.0.1', '--port', '4173', '--strictPort'],
      {
        cwd: frontendRoot,
        stdio: 'ignore',
        windowsHide: true,
      },
    )
    await waitForServer(server)
  }

  const testProcess = spawn(
    process.execPath,
    [playwrightCli, 'test', ...process.argv.slice(2)],
    {
      cwd: frontendRoot,
      stdio: 'inherit',
      windowsHide: true,
    },
  )
  process.exitCode = await waitForExit(testProcess)
} catch (error) {
  console.error(error)
  process.exitCode = 1
} finally {
  if (server) await stopServer(server)
}
