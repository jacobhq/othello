import path from "path"
import tailwindcss from "@tailwindcss/vite"
import {defineConfig} from "vite"
import react from "@vitejs/plugin-react-swc"
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import tanstackRouter from "@tanstack/router-plugin/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    tanstackRouter({
      target: 'react',
      autoCodeSplitting: true,
    }),
    react(),
    tailwindcss(),
    wasm(),
    topLevelAwait()
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      "@wasm": path.resolve(__dirname, "./pkg"),
    },
  },
})
