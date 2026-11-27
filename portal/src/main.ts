import { createApp } from "vue";
import { createPinia } from "pinia";
import { Dialog, Notify, Quasar } from "quasar";
import "@quasar/extras/material-icons/material-icons.css";
import "quasar/src/css/index.sass";
import "./styles.css";

import App from "./App.vue";
import { router } from "./router";
import { initApi } from "./api";
import { useAuthStore } from "./stores/auth";
import { vTarget } from "./lib/targets";

async function boot(): Promise<void> {
  const app = createApp(App);
  const pinia = createPinia();
  app.use(pinia);
  app.directive("target", vTarget);
  app.use(Quasar, {
    plugins: { Notify, Dialog },
    config: { dark: "auto" },
  });
  const auth = useAuthStore();
  await initApi(
    () => auth.token,
    () => auth.rejected(),
  );
  app.use(router);
  app.mount("#app");
}

void boot();
