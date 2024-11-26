<template>
  <div class="mx-auto container py-4 grid grid-cols-1 lg:grid-cols-3 gap-5 md:gap-2 lg:gap-5 min-h-screen" v-if="!isLoading">
    <div class="flex flex-col gap-6 lg:col-span-2">
      <h1 class="font-bold text-2xl">Tổng quan Dataset</h1>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200">
          <p class="font-bold">Số lượng đặc trưng</p>
          <p>{{ overall_info.number_of_feature }}</p>
        </div>
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200">
          <p class="font-bold">Tổng số dữ liệu</p>
          <p>{{ overall_info.number_of_sample }}</p>
        </div>
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200">
          <p class="font-bold">Số lượng dữ liệu huấn luyện</p>
          <p>{{ overall_info.training_sample }}</p>
        </div>
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200">
          <p class="font-bold">Số lượng dữ liệu kiểm thử</p>
          <p>{{ overall_info.testing_sample }}</p>
        </div>
      </div>
      <hr>
      <h1 class="font-bold text-2xl">Các mô hình đã huấn luyện</h1>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div v-for="model in all_model"  :key="model" class="p-4 rounded-xl bg-red-100">
          <div  class="flex justify-between align-middle items-center ">
            <p class="font-bold">Tên mô hình</p>
            <p>{{ model.name.replace("OvO", " - OneVsOne").replace("OvA", " - OneVsAll") }}</p>
          </div>
          <div  class="flex justify-between align-middle items-center ">
            <p class="font-bold">Độ chính xác</p>
            <p>{{ model.accuracy }}%</p>
          </div>
        </div>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-5 gap-4 pb-1">
        <div v-for="(data, index) in non_label_dataset" @click="parseEntry(data)" :key="index" class="p-4 rounded-xl bg-yellow-100 cursor-pointer">
          <div class="flex flex-col justify-between align-middle items-center ">
            <p class="font-bold">Dữ liệu {{ index+1 }}</p>
            <p class="self-start">Pin: {{ data.battery_power}}MiA</p>
            <p class="self-start">RAM: {{ data.ram}}GiB</p>
            <p>(Xem thêm)</p>
          </div>
        </div>
      </div>
    </div>

    <div class="flex flex-col gap-4 lg:pl-4">
      <h1 class="font-bold text-2xl">Dự đoán</h1>
      <DrawComponent class="max-w-20" />
      <button @click="predictNow" class="btn btn-wide mx-auto">Dự đoán ngay</button>
    </div>
    <input type="checkbox" hidden id="my_modal" class="modal-toggle"  />
    <div class="modal" role="dialog">
      <div class="modal-box max-w-fit">
        <div class="flex justify-between">
          <h3 class="text-lg font-bold">Thống kê dự đoán của các mô hình</h3>
          <p class="text-lg font-bold">Kết quả tối ưu: {{ bestChoice }}</p>
        </div>
        <div v-if="predictions.length>0" class="grid grid-cols-4 gap-4 mt-4">
          <div v-for="prediction in predictions" :key="prediction.name" class="p-4 rounded-xl bg-blue-50 shadow-lg">
            <div  class="flex justify-between align-middle items-center gap-4">
              <p class="font-bold">Tên mô hình</p>
              <p>{{ prediction.name.replace("OvO", " - OneVsOne").replace("OvA", " - OneVsAll") }}</p>
            </div>
            <div  class="flex justify-between align-middle items-center gap-4">
              <p class="font-bold">Độ chính xác</p>
              <p>{{ prediction.accuracy }}%</p>
            </div>
            <div  class="flex justify-between align-middle items-center gap-4">
              <p class="font-bold">Kết quả dự đoán</p>
              <p class="font-semibold">{{ prediction.predict_class }}</p>
            </div>
          </div>
        </div>
        <div v-else class="flex flex-col justify-center items-center mt-4">
          <div class="flex items-center justify-center">
            <div class="relative">
              <div class="h-16 w-16 rounded-full border-t-8 border-b-8 border-gray-200"></div>
              <div class="absolute top-0 left-0 h-16 w-16 rounded-full border-t-8 border-b-8 border-blue-500 animate-spin">
              </div>
            </div>
          </div>
          <h1 class="font-bold text-lg">Vui lòng chờ trong giây lát...</h1>
        </div>
      </div>
      <label class="modal-backdrop" for="my_modal">Close</label>
    </div>
  </div>
  <div v-else class="flex items-center justify-center h-screen">
    <div class="relative">
        <div class="h-24 w-24 rounded-full border-t-8 border-b-8 border-gray-200"></div>
        <div class="absolute top-0 left-0 h-24 w-24 rounded-full border-t-8 border-b-8 border-blue-500 animate-spin">
        </div>
    </div>
  </div>
</template>

<script>
import { useExampleStore } from '@/stores/examStore'
import DrawComponent  from '@/pages/home/DrawComponent.vue'
import axios from 'axios';

const api_url = "http://localhost:5000/"

export default {
  name: "home-page",
  components: {DrawComponent},
  setup() {
    const store = useExampleStore()
    return {
      store
    }
  },
  data(){
    return {
      entry: {
        'battery_power': null,
        'blue': false,
        'clock_speed': null,
        'dual_sim': false,
        'fc': null,
        'four_g': false,
        'int_memory': null,
        'm_dep': null,
        'mobile_wt': null,
        'n_cores': null,
        'pc': null,
        'px_height': null,
        'px_width': null,
        'ram': null,
        'sc_h': null,
        'sc_w': null,
        'talk_time': null,
        'three_g': false,
        'touch_screen': false,
        'wifi': false,
      },
      "predictions": [],
      "overall_info": {
        "number_of_feature": "",
        "number_of_sample": "",
        "testing_sample": "",
        "training_sample": ""
      },
      "all_model": [],
      "non_label_dataset": [],
      "isLoading": true
    }
  },
  methods: {
    parseEntry(data) {
      let keys = ["dual_sim", "four_g", "blue", "three_g", "touch_screen", "wifi"]
      for (let key in keys) {
        if (data[keys[key]] == 1) {
          data[keys[key]] = true
        } else {
          data[keys[key]] = false
        }
      }
      this.entry = data
    },
    predictNow() {
      for (let key in this.entry) {
        if (key in this.entry) {
          if (this.entry[key] == null) {
            this.$swal('Vui lòng điền đầy đủ thông tin', '', "error");
            return
          }
        }
      }
      for (let key in this.entry) {
        if (key in this.entry) {
          if (this.entry[key] == true) {
            this.entry[key] = 1
            continue
          }
          if (this.entry[key] == false) {
            this.entry[key] = 0
            continue
          }
        }
      }
      this.predictions = []
      this.bestChoice = ""
      axios.post(api_url + "recognize", this.entry, {timeout: 60000})
        .then(response => {
          const modal = document.getElementById('my_modal')
          modal.click()
          this.predictions =  response.data.entries
          this.bestChoice = response.data.best_choice
          this.entry = {
            'battery_power': null,
            'blue': false,
            'clock_speed': null,
            'dual_sim': false,
            'fc': null,
            'four_g': false,
            'int_memory': null,
            'm_dep': null,
            'mobile_wt': null,
            'n_cores': null,
            'pc': null,
            'px_height': null,
            'px_width': null,
            'ram': null,
            'sc_h': null,
            'sc_w': null,
            'talk_time': null,
            'three_g': false,
            'touch_screen': false,
            'wifi': false,
          }
        })
        .catch(error => {
          console.error(error);
        });
    },
    get_overall_info() {
      axios.get(api_url + "health", {timeout: 60000})
        .then(response => {
          this.overall_info = response.data
        })
        .catch(error => {
          console.error(error);
        });
    },
    get_all_model() {
      axios.get(api_url + "model", {timeout: 60000})
        .then(response => {
          this.all_model = response.data.entries
        })
        .catch(error => {
          console.error(error);
        });
    },
    get_non_label_dataset() {
      axios.get(api_url + "raw-data", {timeout: 60000})
        .then(response => {
          this.non_label_dataset = response.data.entries
        })
        .catch(error => {
          console.error(error);
        });
    }
  },
  async created() {
    await this.get_overall_info()
    // await this.get_all_model()
    // this.get_non_label_dataset()
    this.isLoading = false
  }
};
</script>