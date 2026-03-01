// @ts-check

import react from '@astrojs/react';
import {defineConfig} from 'astro/config';
import topLevelAwait from "vite-plugin-top-level-await";
import wasm from "vite-plugin-wasm";
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
    // Enable React to support React JSX components.
    integrations: [react()],

    vite: {
        plugins: [
            tailwindcss(),
            wasm(),
            topLevelAwait(),
        ],
        optimizeDeps: {
            exclude: ["onnxruntime-web"], // prevent esbuild from mangling ORT's WASM imports
        },
        assetsInclude: ["**/*.onnx"],
    },
});