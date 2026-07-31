import { readFileSync } from 'node:fs'

import { describe, expect, it } from 'vitest'

const tokens = readFileSync('src/styles/tokens.css', 'utf8')

describe('design tokens', () => {
  it('defines the core colors and spacing scale', () => {
    for (const token of [
      '--color-neutral-0:',
      '--color-neutral-50:',
      '--color-text-primary:',
      '--color-action:',
      '--color-success:',
      '--color-warning:',
      '--color-error:',
      '--color-border:',
      '--space-1:',
      '--space-2:',
      '--space-4:',
    ]) {
      expect(tokens).toContain(token)
    }
  })

  it('limits corner radii to 8px and avoids gradients', () => {
    const radii = [...tokens.matchAll(/--radius-[\w-]+:\s*(\d+)px;/g)]

    expect(radii.length).toBeGreaterThan(0)
    for (const radius of radii) {
      expect(Number(radius[1])).toBeLessThanOrEqual(8)
    }
    expect(tokens).not.toMatch(/gradient/i)
  })
})
