import { sentryVitePlugin } from "@sentry/vite-plugin";
import tailwindcss from "@tailwindcss/vite";
import tanstackRouter from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { defineConfig } from "vite";
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    tanstackRouter({
      target: "react",
      autoCodeSplitting: true,
    }),
    react(),
    tailwindcss(),
    wasm(),
    topLevelAwait(),
    sentryVitePlugin({
      org: "jhqcat",
      project: "othello",
    }),
  ],
  optimizeDeps: {
    exclude: ["onnxruntime-web"], // prevent esbuild from mangling ORT's WASM imports
  },
  assetsInclude: ["**/*.onnx"],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@wasm": path.resolve(__dirname, "./pkg"),
    },
  },

  build: {
    sourcemap: true,
  },
});
