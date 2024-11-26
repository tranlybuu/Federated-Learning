<template>
    <nav class="bg-main/60 dark:bg-opmain/60 fixed top-0 w-screen shadow-md backdrop-blur-xl text-gray-700 hover:text-gray-500 dark:text-white">
        <div class="mx-auto container px-4 md:px-0">
          <div class="relative flex h-16 flex items-center justify-between">
            <!-- Logo and Desktop menu -->
            <div class="flex flex-1 items-center md:items-stretch justify-start">
              <div class="flex flex-shrink-0 items-center gap-5">
                <img class="h-8 w-auto drop-shadow-sm" src="../assets/img/logo.png" alt="Your Company">
                <p class="font-bold text-lg md:text-xl hidden md:block">Khoa Kỹ Thuật và Công Nghệ</p>
              </div>
            </div>

            <p class="font-bold text-xl">Học liên kết & Ứng dụng</p>
          </div>
        </div>
      
        <!-- Mobile menu, show/hide based on menu state. -->
        <div v-show="isOpenMenu" class="md:hidden border-t-[1px] border-gray-300 dark:border-gray-800" id="mobile-menu">
          <div class="space-y-1 px-2 pb-3 pt-2">
            <div v-for="(item, index) in menu" :key="'menu'+index">
                <a v-if="item.path == currentPage" href="#" class="text-red-800 dark:text-red-400 block rounded-md px-3 py-2 text-base font-medium" aria-current="page">{{item.name}}</a>
                <a v-else href="#" class="text-black dark:text-white hover:text-red-900 dark:hover:text-red-300  block rounded-md px-3 py-2 text-base font-medium">{{item.name}}</a>
            </div>
          </div>
        </div>
      </nav>
</template>

<script>
export default {
    name: 'Header',
    data() {
        return {
            isOpenMenu: false,
            isOpenUserProfile: false,
            isDarkmode: false
        }
    }
}
</script>