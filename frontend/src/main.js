import { createApp } from "vue";
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import VueSweetalert2 from 'vue-sweetalert2';
import VuePageTransition from 'vue-page-transition'
import VueViewer from 'v-viewer'

// Css
import './assets/css/style.css'
import 'sweetalert2/dist/sweetalert2.min.css';
import 'viewerjs/dist/viewer.css'
import 'vue-multiselect/dist/vue-multiselect.css'
import "@egjs/vue3-flicking/dist/flicking.css";
import "@egjs/vue3-flicking/dist/flicking-inline.css";
import "@egjs/flicking-plugins/dist/flicking-plugins.css";

const app = createApp(App);
app.use(VueSweetalert2);
app.use(VuePageTransition);
app.use(VueViewer);
app.use(router);
app.use(createPinia());
app.mount("#app");