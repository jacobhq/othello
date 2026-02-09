use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("../othello-wasm/src/model/demo.onnx")
        .out_dir("model/")
        .embed_states(true)  // Need embedded states in this crate to convert them
        .run_from_script();
}
