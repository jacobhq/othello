use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/demo.onnx")
        .out_dir("model/")
        .embed_states(true)
        .run_from_script();
}
