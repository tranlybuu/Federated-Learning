import { defineStore } from 'pinia'
import axios from 'axios';

const api_url = "http://localhost:8000/api/"

export const useExampleStore = defineStore('example', {
    state: () => ({ 
        // count: 0,
    }),
    getters: {
        // doubleCount: (state) => state.count * 2,
    },
    actions: {
        // increment() {
        //     this.count++
        // },
        // decrement() {
        //     this.count--
        // },
        // resetCount() {
        //     this.count = 0
        // },
        get_overall_info() {
            axios.get(api_url + "info")
              .then(response => {
                return response
              })
              .catch(error => {
                console.error(error);
              });
        }        
    },
})