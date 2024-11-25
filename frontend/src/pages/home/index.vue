<template>
  <div class="mx-auto container py-4 grid grid-cols-1 lg:grid-cols-2 gap-5 md:gap-2 lg:gap-5 min-h-screen" v-if="!isLoading">
    <div class="flex flex-col gap-6">
      <h1 class="font-bold text-2xl">Tổng quan Dataset</h1>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200">
          <p class="font-bold">Số lượng đặc trưng</p>
          <p>{{ overall_info.number_of_feature }}</p>
        </div>
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200">
          <p class="font-bold">Tổng số bảng ghi</p>
          <p>{{ overall_info.number_of_sample }}</p>
        </div>
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200">
          <p class="font-bold">Số lượng bảng ghi huấn luyện</p>
          <p>{{ overall_info.training_sample }}</p>
        </div>
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200">
          <p class="font-bold">Số lượng bảng ghi kiểm thử</p>
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
      <hr>
      <div class="flex justify-between">
        <h1 class="font-bold text-2xl">Dữ liệu ngẫu nhiên chưa phân loại</h1>
        <button @click="get_non_label_dataset" class="btn btn-square hover:rotate-90">
          <svg xmlns="http://www.w3.org/2000/svg" height="16" width="16" viewBox="0 0 512 512"><path d="M142.9 142.9c62.2-62.2 162.7-62.5 225.3-1L327 183c-6.9 6.9-8.9 17.2-5.2 26.2s12.5 14.8 22.2 14.8H463.5c0 0 0 0 0 0H472c13.3 0 24-10.7 24-24V72c0-9.7-5.8-18.5-14.8-22.2s-19.3-1.7-26.2 5.2L413.4 96.6c-87.6-86.5-228.7-86.2-315.8 1C73.2 122 55.6 150.7 44.8 181.4c-5.9 16.7 2.9 34.9 19.5 40.8s34.9-2.9 40.8-19.5c7.7-21.8 20.2-42.3 37.8-59.8zM16 312v7.6 .7V440c0 9.7 5.8 18.5 14.8 22.2s19.3 1.7 26.2-5.2l41.6-41.6c87.6 86.5 228.7 86.2 315.8-1c24.4-24.4 42.1-53.1 52.9-83.7c5.9-16.7-2.9-34.9-19.5-40.8s-34.9 2.9-40.8 19.5c-7.7 21.8-20.2 42.3-37.8 59.8c-62.2 62.2-162.7 62.5-225.3 1L185 329c6.9-6.9 8.9-17.2 5.2-26.2s-12.5-14.8-22.2-14.8H48.4h-.7H40c-13.3 0-24 10.7-24 24z"/></svg>
        </button>
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
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Tổng dung lượng pin của máy hiện tại</span>
          </div>
          <input type="number" placeholder="battery_power" v-model="entry.battery_power"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Tốc độ bộ vi xử lý (CPU)</span>
          </div>
          <input type="number" placeholder="clock_speed" v-model="entry.clock_speed"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Độ phân giải của camera trước</span>
          </div>
          <input type="number" placeholder="fc" v-model="entry.fc"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Dung lượng bộ nhớ ROM</span>
          </div>
          <input type="number" placeholder="int_memory" v-model="entry.int_memory"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Độ dày của máy</span>
          </div>
          <input type="number" placeholder="m_dep" v-model="entry.m_dep"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Cân nặng của máy</span>
          </div>
          <input type="number" placeholder="mobile_wt" v-model="entry.mobile_wt"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Số nhân xử lý của cpu</span>
          </div>
          <input type="number" placeholder="n_cores" v-model="entry.n_cores"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Độ phân giải camera chính</span>
          </div>
          <input type="number" placeholder="pc" v-model="entry.pc"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Độ phân giải theo chiều cao màn hình</span>
          </div>
          <input type="number" placeholder="px_height" v-model="entry.px_height"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Độ phân giải theo chiều rộng màn hình</span>
          </div>
          <input type="number" placeholder="px_width" v-model="entry.px_width"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Dung lượng bộ nhớ RAM</span>
          </div>
          <input type="number" placeholder="ram" v-model="entry.ram"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Chiều cao của màn hình</span>
          </div>
          <input type="number" placeholder="sc_h" v-model="entry.sc_h"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Chiều rộng của màn hình</span>
          </div>
          <input type="number" placeholder="sc_w" v-model="entry.sc_w"  class="input input-bordered w-full" />
        </label>
        <label class="form-control w-full">
          <div class="label">
            <span class="label-text">Thời gian sử dụng lâu nhất</span>
          </div>
          <input type="number" placeholder="talk_time" v-model="entry.talk_time"  class="input input-bordered w-full" />
        </label>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-3">
          <label class="form-control w-full">
            <div class="label">
              <span class="label-text">Màn hình cảm ứng</span>
            </div>
            <input type="checkbox" class="toggle toggle-primary"  v-model="entry.touch_screen" />
          </label>
          <label class="form-control w-full">
            <div class="label">
              <span class="label-text">Kết nối Wifi</span>
            </div>
            <input type="checkbox" class="toggle toggle-primary"  v-model="entry.wifi"/>
          </label>
          <label class="form-control w-full">
            <div class="label">
              <span class="label-text">Kết nối bluetooth</span>
            </div>
            <input type="checkbox" class="toggle toggle-primary" v-model="entry.blue" />
          </label>
          <label class="form-control w-full">
            <div class="label">
              <span class="label-text">Sử dụng được 2 sim</span>
            </div>
            <input type="checkbox" class="toggle toggle-primary"  v-model="entry.dual_sim" />
          </label>
          <label class="form-control w-full">
            <div class="label">
              <span class="label-text">Kết nối 3G</span>
            </div>
            <input type="checkbox" class="toggle toggle-primary"  v-model="entry.three_g"/>
          </label>
          <label class="form-control w-full">
            <div class="label">
              <span class="label-text">Kết nối 4G</span>
            </div>
            <input type="checkbox" class="toggle toggle-primary"  v-model="entry.four_g"/>
          </label>
        </div>
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
import axios from 'axios';

const api_url = "http://localhost:8000/api/"

export default {
  name: "home-page",
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
      axios.post(api_url + "predict", this.entry, {timeout: 60000})
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
      axios.get(api_url + "info", {timeout: 60000})
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
          this.isLoading = false
        })
        .catch(error => {
          console.error(error);
        });
    }
  },
  async created() {
    await this.get_overall_info()
    await this.get_all_model()
    this.get_non_label_dataset()
  }
};
</script>