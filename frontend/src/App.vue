<script setup lang="ts">
import { computed } from 'vue'
import { RouterView, useRoute } from 'vue-router'

import AppShell from './app/AppShell.vue'

const route = useRoute()
const usesShell = computed(() => Boolean(route.meta.requiresAuth))
</script>

<template>
  <AppShell v-if="usesShell" />
  <RouterView v-else />
</template>
