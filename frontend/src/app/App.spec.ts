import { render, screen } from '@testing-library/vue'
import { createMemoryHistory, createRouter } from 'vue-router'
import { describe, expect, it } from 'vitest'
import App from '../App.vue'

describe('App', () => {
  it('renders the matched route through RouterView', async () => {
    const router = createRouter({
      history: createMemoryHistory(),
      routes: [
        {
          path: '/',
          component: {
            template: '<main data-testid="route-content">工作台</main>',
          },
        },
      ],
    })

    await router.push('/')
    await router.isReady()

    render(App, { global: { plugins: [router] } })

    expect((await screen.findByTestId('route-content')).textContent).toBe(
      '工作台',
    )
  })
})
