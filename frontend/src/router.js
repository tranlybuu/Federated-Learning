import { createWebHistory, createRouter } from "vue-router";

const routes =  [
  {
    path: "/",
    name: "layout",
    component: () => import("./pages"),
    redirect: {
      name: "home-page",
    },
    children: [
      {
        path: "/",
        name: "home-page",
        component: () => import("./pages/home")
      },
    ]
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;