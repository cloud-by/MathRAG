import { createApp } from 'vue'

import App from './App.vue'
import { authController, authKey } from './features/auth/useAuth'
import router from './router'
import 'katex/dist/katex.min.css'
import './styles/base.css'
import './styles/tokens.css'

const app = createApp(App)

app.provide(authKey, authController)
app.use(router)
app.mount('#app')
