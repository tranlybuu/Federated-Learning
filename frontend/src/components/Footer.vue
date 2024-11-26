<template>
    <div class="py-4 bg-gray-100 shadow-xl">
      <div class="mx-auto container">
        <p>footer</p>
        <p>{{env}}</p>
      </div>
    </div>
</template>

<script>
export default {
    name: 'Footer',
    data() {
        return {
            env: "none"
        }
    },
    methods: {
      
    },
    created() {
      this.env = process.env.VUE_APP_VAR_ENV
      console.log(process)
    }
}
</script>