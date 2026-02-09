use burn_onnx::ModelGen;

fn main() {
    ModelGen::new()
        .input("src/model/demo.onnx")
        .out_dir("model/")
        .embed_states(false)  // Don't embed - we'll load via BinBytesRecorder
        .run_from_script();
}
