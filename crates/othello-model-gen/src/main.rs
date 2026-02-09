//! Convert ONNX model weights from burnpack format to BinBytesRecorder format.
//! 
//! This allows the model to be loaded in WASM without condvar issues.

use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

mod model {
    include!(concat!(env!("OUT_DIR"), "/model/demo.rs"));
}

use model::Model;

fn main() {
    println!("Loading model from embedded burnpack...");
    
    // Load model with weights from burnpack (generated at build time)
    let device = Default::default();
    let model: Model<NdArray<f32>> = Model::from_embedded(&device);
    
    println!("Extracting model record...");
    let record = model.into_record();
    
    println!("Saving to model.bin using BinBytesRecorder...");
    let recorder = BinBytesRecorder::<FullPrecisionSettings>::default();
    
    // Save to file
    let bytes = recorder
        .record(record, ())
        .expect("Failed to encode model record");
    
    let output_path = std::path::Path::new("../othello-wasm/model.bin");
    std::fs::write(output_path, bytes).expect("Failed to write model.bin");
    
    println!("Successfully saved model.bin ({} bytes)", 
        std::fs::metadata(output_path).unwrap().len());
}
