import { fileURLToPath, URL } from "node:url";
import { defineConfig, loadEnv } from "vite";
import vue from "@vitejs/plugin-vue";
import { quasar, transformAssetUrls } from "@quasar/vite-plugin";

// pawn-server serves the build at /portal (ApiSection.portal_dir → portal/dist).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.PAWN_API_URL || "http://localhost:8000";
  return {
    base: "/portal/",
    plugins: [vue({ template: { transformAssetUrls } }), quasar()],
    resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
    build: { target: "es2022", outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
    server: {
      port: 5173,
      proxy: {
        "/v1": { target, changeOrigin: true },
        "/health": { target, changeOrigin: true },
      },
    },
  };
});
