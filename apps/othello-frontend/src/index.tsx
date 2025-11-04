import {StrictMode} from 'react'
import {createRoot} from 'react-dom/client'
import {Toaster} from "@/components/ui/sonner"
import {PostHogProvider} from "posthog-js/react";
import './index.css'
import App from './App.tsx'

const options = {
    api_host: import.meta.env.VITE_PUBLIC_POSTHOG_HOST,
    defaults: "2025-05-24",
} as const

createRoot(document.getElementById('root')!).render(
    <StrictMode>
        <PostHogProvider apiKey={import.meta.env.VITE_PUBLIC_POSTHOG_KEY} options={options}>
            <App/>
            <Toaster/>
        </PostHogProvider>
    </StrictMode>,
)
