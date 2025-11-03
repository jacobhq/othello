import Hero from "@/components/marketing/hero";
import Board from "@/components/game/board.tsx";

function App() {

    return (
        <div className="p-8">
            <Hero />
            <div>
                <Board />
            </div>
        </div>
    );
}

export default App;
