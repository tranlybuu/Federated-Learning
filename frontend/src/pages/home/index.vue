<template>
  <div class="mx-auto container py-4 grid grid-cols-1 lg:grid-cols-3 gap-5 md:gap-2 lg:gap-3 h-max" v-if="!isLoading">
    <div class="flex flex-col gap-4 lg:col-span-2">
      <h1 class="font-bold text-2xl">Tổng quan Dataset</h1>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200 shadow-sm shadow-gray-400">
          <p class="font-bold">Số lượng đặc trưng</p>
          <p>28 x 28 = <b>784</b></p>
        </div>
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200 shadow-sm shadow-gray-400">
          <p class="font-bold">Dữ liệu phân loại</p>
          <p><b>{{ overall_info.client_labels.length }} lớp</b> [{{ overall_info.client_labels.sort().join(', ') }}]</p>
        </div>
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200 shadow-sm shadow-gray-400">
          <p class="font-bold">Tổng số dữ liệu</p>
          <p>{{ overall_info.dataset_size.train + overall_info.dataset_size.test }}</p>
        </div>
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200 shadow-sm shadow-gray-400">
          <p class="font-bold">Số lượng dữ liệu huấn luyện</p>
          <p>{{ overall_info.dataset_size.train }}</p>
        </div>
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200 shadow-sm shadow-gray-400">
          <p class="font-bold">Số lượng dữ liệu kiểm thử</p>
          <p>{{ overall_info.dataset_size.test }}</p>
        </div>
        <div class="flex justify-between align-middle items-center p-4 rounded-xl bg-gray-200 shadow-sm shadow-gray-400">
          <p class="font-bold">Số lượng vòng huấn luyện</p>
          <p>{{ overall_info.total_rounds }}</p>
        </div>
      </div>
      <AccuracyChart :accuracyHistory="overall_info.accuracy_history" />
      <hr>
      <h1 class="font-bold text-2xl">Các mô hình đã huấn luyện</h1>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div v-for="model in all_model"  :key="model.path" >
          <div @click="change_current_model(model.name)" class="p-4 rounded-xl border-[2px] ease-in-out duration-200"
              :class="[model.name == choosing_model ? 'bg-red-300 border-black' : 'bg-gray-100 border-gray-400']">
            <div  class="flex justify-between align-middle items-center">
              <p class="font-bold">Tên mô hình</p>
              <p>{{ model.name.replace(".keras", "") }}</p>
            </div>
            <div  class="flex justify-between align-middle items-center ">
              <p class="font-bold">Thời gian cập nhật</p>
              <p>{{ model.last_modified }}</p>
            </div>
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
      <h1 class="font-bold text-2xl text-center">Dự đoán</h1>
      <DrawComponent @update="v => predictNow(v)" />
    </div>
    <input type="checkbox" hidden id="my_modal" class="modal-toggle"  />
    <div class="modal" role="dialog">
      <div class="modal-box max-w-fit">
        <div v-if="prediction != null">
          <div v-if="prediction.success == true" class="flex flex-col md:flex-row gap-4 md:gap-8 w-auto">
            <div class="flex flex-col gap-4 justify-between">
              <div class="flex flex-col gap-4 items-start">
                <h3 class="text-2xl font-extrabold text-red-900 italic lg:w-max">Thống kê dự đoán của mô hình</h3>
                <div  class="flex justify-between align-middle items-center gap-4">
                  <p class="font-bold">Tên mô hình</p>
                  <p>{{ prediction.model_info.name }}</p>
                </div>
                <div  class="flex justify-between align-middle items-center gap-4">
                  <p class="font-bold">Mức độ tự tin</p>
                  <p>{{ prediction.confidence }}%</p>
                </div>
                <div  class="flex justify-between align-middle items-center gap-4">
                  <p class="font-bold">Kết quả dự đoán</p>
                  <p class="font-semibold">{{ prediction.digit }}</p>
                </div>
              </div>
              <div  class="flex justify-between align-middle items-center gap-4">
                <p class="font-bold">Thời gian dự đoán</p>
                <p>{{ prediction.prediction_time }}</p>
              </div>
            </div>
            <div class="relative w-full md:max-w-[250px] mx-auto bg-gray-100">
              <div class="grid grid-cols-3">
                <!-- Row 1 -->
                <div class="rounded-tl-lg flex flex-col items-center justify-center py-3 m-[0.5px] border-[1px] min-w-[80px]" 
                    :class="[1 == prediction.digit ? 'bg-gray-300 border-black' : 'bg-gray-200 border-gray-400']">
                  <span class="text-2xl font-bold">1</span>
                  <span class="text-sm mt-0.5 text-gray-600">{{ prediction.all_confidence[1] }}</span>
                </div>
                <div class="flex flex-col items-center justify-center py-3 m-[0.5px] border-[1px] min-w-[80px]" 
                    :class="[2 == prediction.digit ? 'bg-gray-300 border-black' : 'bg-gray-200 border-gray-400']">
                  <span class="text-2xl font-bold">2</span>
                  <span class="text-sm mt-0.5 text-gray-600">{{ prediction.all_confidence[2] }}</span>
                </div>
                <div class="rounded-tr-lg flex flex-col items-center justify-center py-3 m-[0.5px] border-[1px] min-w-[80px]" 
                    :class="[3 == prediction.digit ? 'bg-gray-300 border-black' : 'bg-gray-200 border-gray-400']">
                  <span class="text-2xl font-bold">3</span>
                  <span class="text-sm mt-0.5 text-gray-600">{{ prediction.all_confidence[3] }}</span>
                </div>

                <!-- Row 2 -->
                <div class="flex flex-col items-center justify-center py-3 m-[0.5px] border-[1px] min-w-[80px]" 
                    :class="[4 == prediction.digit ? 'bg-gray-300 border-black' : 'bg-gray-200 border-gray-400']">
                  <span class="text-2xl font-bold">4</span>
                  <span class="text-sm mt-0.5 text-gray-600">{{ prediction.all_confidence[4] }}</span>
                </div>
                <div class="flex flex-col items-center justify-center py-3 m-[0.5px] border-[1px] min-w-[80px]" 
                    :class="[5 == prediction.digit ? 'bg-gray-300 border-black' : 'bg-gray-200 border-gray-400']">
                  <span class="text-2xl font-bold">5</span>
                  <span class="text-sm mt-0.5 text-gray-600">{{ prediction.all_confidence[5] }}</span>
                </div>
                <div class="flex flex-col items-center justify-center py-3 m-[0.5px] border-[1px] min-w-[80px]" 
                    :class="[6 == prediction.digit ? 'bg-gray-300 border-black' : 'bg-gray-200 border-gray-400']">
                  <span class="text-2xl font-bold">6</span>
                  <span class="text-sm mt-0.5 text-gray-600">{{ prediction.all_confidence[6] }}</span>
                </div>

                <!-- Row 3 -->
                <div class="flex flex-col items-center justify-center py-3 m-[0.5px] border-[1px] min-w-[80px]" 
                    :class="[7 == prediction.digit ? 'bg-gray-300 border-black' : 'bg-gray-200 border-gray-400']">
                  <span class="text-2xl font-bold">7</span>
                  <span class="text-sm mt-0.5 text-gray-600">{{ prediction.all_confidence[7] }}</span>
                </div>
                <div class="flex flex-col items-center justify-center py-3 m-[0.5px] border-[1px] min-w-[80px]" 
                    :class="[8 == prediction.digit ? 'bg-gray-300 border-black' : 'bg-gray-200 border-gray-400']">
                  <span class="text-2xl font-bold">8</span>
                  <span class="text-sm mt-0.5 text-gray-600">{{ prediction.all_confidence[8] }}</span>
                </div>
                <div class="flex flex-col items-center justify-center py-3 m-[0.5px] border-[1px] min-w-[80px]" 
                    :class="[9 == prediction.digit ? 'bg-gray-300 border-black' : 'bg-gray-200 border-gray-400']">
                  <span class="text-2xl font-bold">9</span>
                  <span class="text-sm mt-0.5 text-gray-600">{{ prediction.all_confidence[9] }}</span>
                </div>

                <!-- Row 4 -->
                <div class="rounded-b-lg flex flex-col col-span-3 items-center justify-center py-3 m-[0.5px] border-[1px] min-w-[80px]" 
                    :class="[0 == prediction.digit ? 'bg-gray-300 border-black' : 'bg-gray-200 border-gray-400']">
                  <span class="text-2xl font-bold">0</span>
                  <span class="text-sm mt-0.5 text-gray-600">{{ prediction.all_confidence[0] }}</span>
                </div>
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
            <h3 class="text-2xl font-extrabold text-red-900 italic lg:w-max">Phát hiện lỗi xảy ra</h3>
            <h1 class="font-bold text-lg">{{ prediction.error }}</h1>
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
import AccuracyChart  from '@/pages/home/AccuracyChart.vue'
import axios from 'axios';

export default {
  name: "home-page",
  components: {DrawComponent, AccuracyChart},
  setup() {
    const store = useExampleStore()
    return {
      store
    }
  },
  data(){
    return {
      "api_url": "http://localhost:5000/",
      "prediction": null,
      "overall_info": {},
      "all_model": [],
      "choosing_model": "",
      "isLoading": true,
      "qr_image": ""
    }
  },
  methods: {
    predictNow(imageUrl) {
      this.prediction =  null
      const modal = document.getElementById('my_modal')
      modal.click()
      axios.post(this.api_url + "recognize", imageUrl, {
        params: {
          model: this.choosing_model,
        },
        headers: {
          'Content-Type': 'image/png'
        },
        timeout: 60000
      })
        .then(response => {
          this.prediction =  response.data
        })
        .catch(error => {
          console.log("ERROR =>", error)
        });
    },
    async get_overall_info() {
      await axios.get(this.api_url + "model-stats/" + this.choosing_model, {timeout: 60000})
        .then(response => {
          this.overall_info = response.data
        })
        .catch(error => {
          console.error(error);
        });
    },
    async get_all_model() {
      await axios.get(this.api_url + "health", {timeout: 60000})
        .then(response => {
          this.all_model = []
          for (let index in response.data.available_models) {
            let model = response.data.available_models[index]
            if (model.name.includes("best_") || model.name.includes("initial")) {
              this.all_model.push(model)
            }
          }
          this.choosing_model = this.all_model[0]?.name ? this.all_model[0].name : ""
        })
        .catch(error => {
          console.error(error);
        });
    },
    change_current_model(name) {
      this.choosing_model = name
      this.get_overall_info()
    },
    get_non_label_dataset() {
      axios.get(this.api_url + "raw-data", {timeout: 60000})
        .then(response => {
          this.non_label_dataset = response.data.entries
        })
        .catch(error => {
          console.error(error);
        });
    },
  },
  async created() {
    this.api_url = window.location
    await this.get_all_model()
    await this.get_overall_info()
    // this.get_non_label_dataset()
    this.isLoading = false
  }
};
</script>