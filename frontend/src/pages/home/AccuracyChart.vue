// AccuracyCards.vue
<template>
  <div class="w-full mx-auto">
    <div class="flex space-x-6 overflow-x-auto pb-2 pr-1">
      <!-- Map qua từng round -->
      <div 
        v-for="round in accuracyHistory" 
        :key="round.round"
        class="flex-none w-64 bg-gray-100 rounded-xl shadow-sm shadow-gray-400 p-6"
      >
        <!-- Round header -->
        <div class="text-center mb-4">
          <h3 class="text-lg font-bold">Round {{ round.round }}</h3>
          <p class="text-2xl font-bold text-red-600 mt-1">
            {{ formatAccuracy(round.accuracy) }}
          </p>
        </div>

        <!-- Client accuracies -->
        <div class="space-y-2">
          <div 
            v-for="(accuracy, clientId) in round.client_accuracies" 
            :key="clientId"
            class="flex justify-between items-center px-2 py-1 bg-gray-50 rounded-lg shadow-md hover:bg-gray-100 transition-colors"
          >
            <span class="text-sm font-medium">
              Client {{ clientId }}:
            </span>
            <span class="text-sm text-red-600">
              {{ formatAccuracy(accuracy) }}
            </span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'AccuracyCards',
  
  props: {
    accuracyHistory: {
      type: Array,
      required: true,
      default: () => []
    }
  },

  methods: {
    formatAccuracy(value) {
      return Number(value).toFixed(4)
    }
  }
}
</script>

<style scoped>
.overflow-x-auto {
  -webkit-overflow-scrolling: touch;
  scrollbar-width: thin;
}

.overflow-x-auto::-webkit-scrollbar {
  height: 6px;
}

.overflow-x-auto::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

.overflow-x-auto::-webkit-scrollbar-thumb {
  background: #cbd5e0;
  border-radius: 3px;
}

.overflow-x-auto::-webkit-scrollbar-thumb:hover {
  background: #a0aec0;
}
</style>