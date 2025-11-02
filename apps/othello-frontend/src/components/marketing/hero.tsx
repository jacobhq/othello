import { Button } from '@/components/ui/button.tsx'

export default function Hero() {
    return (
        <section className="py-20">
            <div className="relative z-10 mx-auto w-full max-w-2xl px-6 lg:px-0">
                <div className="relative text-center">
                    <h1 className="mx-auto mt-16 max-w-xl text-balance text-5xl font-medium font-serif">Othello</h1>

                    <p className="text-muted-foreground mx-auto mb-6 mt-4 text-balance text-xl font-serif">Become extraordinarily focused with a strategy game that is elegant, simple, and fun.</p>

                    <div className="flex flex-col items-center gap-2 *:w-full sm:flex-row sm:justify-center sm:*:w-auto">
                        <Button
                            rounded="full"
                            asChild>
                            <a href="#link">
                                <span className="text-nowrap">Get Started</span>
                            </a>
                        </Button>
                        <Button
                            rounded="full"
                            asChild
                            variant="ghost">
                            <a href="#link">
                                <span className="text-nowrap">View Demo</span>
                            </a>
                        </Button>
                    </div>
                </div>

                <div className="relative mt-12 overflow-hidden rounded-3xl bg-black/10 md:mt-16">
                    <img
                        src="https://images.unsplash.com/photo-1547623641-d2c56c03e2a7?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
                        alt=""
                        className="absolute inset-0 size-full object-cover"
                    />

                    <div className="bg-background rounded-(--radius) relative m-4 overflow-hidden border border-transparent shadow-xl shadow-black/15 ring-1 ring-black/10 sm:m-8 md:m-12">
                        <img
                            src="/mist/tailark-2.png"
                            alt="app screen"
                            width="2880"
                            height="1842"
                            className="object-top-left size-full object-cover"
                        />
                    </div>
                </div>

                <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
                    <p className="text-muted-foreground text-center font-serif">&copy; 2025 Jacob Marshall</p>
                </div>
            </div>
        </section>
    )
}
