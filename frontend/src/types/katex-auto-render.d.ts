declare module 'katex/contrib/auto-render' {
  import type { KatexOptions } from 'katex'

  interface Delimiter {
    display: boolean
    left: string
    right: string
  }

  interface AutoRenderOptions extends KatexOptions {
    delimiters?: Delimiter[]
    ignoredClasses?: string[]
    ignoredTags?: string[]
    preProcess?: (math: string) => string
  }

  export default function renderMathInElement(
    element: HTMLElement,
    options?: AutoRenderOptions,
  ): void
}
