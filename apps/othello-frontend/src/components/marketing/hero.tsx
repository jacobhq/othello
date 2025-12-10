import { Button } from '@/components/ui/button.tsx'
import {Skeleton} from "@/components/ui/skeleton.tsx";
import {Link} from "@tanstack/react-router";

export default function Hero() {
    return (
        <section className="py-16">
            <div className="relative z-10 mx-auto w-full max-w-2xl px-6 lg:px-0">
                <div className="relative text-center">
                    <h1 className="mx-auto mt-16 max-w-xl text-balance text-5xl font-medium font-serif">Othello</h1>

                    <p className="text-muted-foreground mx-auto mb-6 mt-4 text-balance text-xl font-serif">Become extraordinarily focused with a strategy game that is elegant, simple, and fun.</p>

                    <div className="flex flex-col items-center gap-2 *:w-full sm:flex-row sm:justify-center sm:*:w-auto">
                        <Button
                            rounded="full"
                            asChild>
                            <a href="#game">
                                <span className="text-nowrap">Play Othello</span>
                            </a>
                        </Button>
                        <Button
                            rounded="full"
                            asChild
                            variant="ghost">
                            <Link to="/auth/signup">
                                <span className="text-nowrap">Sign Up</span>
                            </Link>
                        </Button>
                    </div>
                </div>

                <div className="relative mt-12 overflow-hidden rounded-3xl bg-black/10 md:mt-16">
                    <img
                        src="https://images.unsplash.com/photo-1547623641-d2c56c03e2a7?q=80&w=3087&auto=format&fit=crop"
                        alt=""
                        className="absolute inset-0 size-full object-cover"
                    />

                    <div className="bg-white opacity-50 rounded-lg relative m-4 overflow-hidden border border-transparent shadow-xl shadow-black/15 ring-1 ring-black/10 sm:m-8 md:m-12 aspect-square">
                        <div className="grid grid-rows-8 rounded-md p-1 sm:p-2 xl:p-3 gap-1 sm:gap-2 xl:gap-3">
                            {Array.from({length: 8}, (_, i) => (
                                <div className="grid grid-cols-8 gap-1 sm:gap-2 xl:gap-3" key={i}>
                                    {Array.from({length: 8}, (_, j) => (
                                        <Skeleton className="bg-neutral-200 rounded-sm sm:rounded p-1 sm:p-2 xl:p3 aspect-square" key={j}>

                                        </Skeleton>
                                    ))}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
                    <p className="text-muted-foreground text-center font-serif">&copy; 2025 Jacob Marshall</p>
                </div>
            </div>
        </section>
    )
}
