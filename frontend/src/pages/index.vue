<script>
import Header from "@/components/Header.vue"
export default {
  components: {Header}
    
}
</script>

<template>
    <div class="pt-14 bg-main dark:bg-opmain text-opmain dark:text-main min-h-screen overflow-auto">
      <Header />
      <router-view class="px-4 md:px-2" />
    </div>
  </template>
  