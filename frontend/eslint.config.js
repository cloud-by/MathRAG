import js from '@eslint/js'
import pluginVue from 'eslint-plugin-vue'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  {
    ignores: ['node_modules', 'dist', 'coverage', '**/*.d.ts'],
  },
  js.configs.recommended,
  tseslint.configs.recommended,
  ...pluginVue.configs['flat/essential'],
  {
    files: ['scripts/**/*.mjs'],
    languageOptions: {
      globals: {
        AbortSignal: 'readonly',
        console: 'readonly',
        fetch: 'readonly',
        process: 'readonly',
        setTimeout: 'readonly',
        URL: 'readonly',
      },
    },
  },
  {
    files: ['**/*.vue'],
    languageOptions: {
      globals: {
        AbortController: 'readonly',
        BeforeUnloadEvent: 'readonly',
        document: 'readonly',
        DOMException: 'readonly',
        Event: 'readonly',
        File: 'readonly',
        HTMLButtonElement: 'readonly',
        HTMLInputElement: 'readonly',
        HTMLSelectElement: 'readonly',
        HTMLTextAreaElement: 'readonly',
        HTMLElement: 'readonly',
        KeyboardEvent: 'readonly',
        Node: 'readonly',
        PointerEvent: 'readonly',
        window: 'readonly',
      },
      parserOptions: {
        parser: tseslint.parser,
      },
    },
  },
)
