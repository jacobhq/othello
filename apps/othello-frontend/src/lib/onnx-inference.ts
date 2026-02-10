import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;

/**
 * Create an evaluator function that can be passed to WasmGame.set_evaluator().
 * Loads the ONNX model and returns a function that runs inference.
 */
export async function createEvaluator(): Promise<
    (input: Float32Array) => Promise<{ policy: Float32Array; value: number }>
> {
    if (!session) {
        // Load ORT runtime files from CDN (Vite blocks dynamic imports from public/)
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.1/dist/";

        // Try WebGPU first, fall back to WASM
        const eps: ort.InferenceSession.ExecutionProviderConfig[] = [];

        if ("gpu" in navigator) {
            console.log("[ORT] WebGPU available, trying webgpu EP");
            eps.push("webgpu");
        }
        eps.push("wasm");

        session = await ort.InferenceSession.create("/demo.onnx", {
            executionProviders: eps,
        });

        console.log("[ORT] Session created");
    }

    const sess = session;

    return async (input: Float32Array) => {
        // Input shape: [1, 2, 8, 8]
        const inputTensor = new ort.Tensor("float32", input, [1, 2, 8, 8]);

        const feeds: Record<string, ort.Tensor> = {};
        const inputName = sess.inputNames[0];
        feeds[inputName] = inputTensor;

        const results = await sess.run(feeds);

        // Get output names from the session
        const outputNames = sess.outputNames;
        const policyOutput = results[outputNames[0]];
        const valueOutput = results[outputNames[1]];

        const policy = policyOutput.data as Float32Array;
        const value = (valueOutput.data as Float32Array)[0];

        return { policy: new Float32Array(policy), value };
    };
}

/**
 * Create a batch evaluator function that can be passed to WasmGame.set_batch_evaluator().
 * Runs batched ONNX inference: accepts a flat Float32Array of concatenated board states.
 *
 * Input:  Float32Array of length batchSize * 128 (each board = 2×8×8)
 * Output: { policies: Float32Array(batchSize * 64), values: Float32Array(batchSize) }
 */
export async function createBatchEvaluator(): Promise<
    (inputs: Float32Array, batchSize: number) => Promise<{ policies: Float32Array; values: Float32Array }>
> {
    if (!session) {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.1/dist/";

        const eps: ort.InferenceSession.ExecutionProviderConfig[] = [];
        if ("gpu" in navigator) {
            console.log("[ORT] WebGPU available, trying webgpu EP");
            eps.push("webgpu");
        }
        eps.push("wasm");

        session = await ort.InferenceSession.create("/demo.onnx", {
            executionProviders: eps,
        });

        console.log("[ORT] Session created (batch)");
    }

    const sess = session;

    return async (inputs: Float32Array, batchSize: number) => {
        const inputTensor = new ort.Tensor("float32", inputs, [batchSize, 2, 8, 8]);

        const feeds: Record<string, ort.Tensor> = {};
        const inputName = sess.inputNames[0];
        feeds[inputName] = inputTensor;

        const results = await sess.run(feeds);

        const outputNames = sess.outputNames;
        const policyOutput = results[outputNames[0]];
        const valueOutput = results[outputNames[1]];

        return {
            policies: new Float32Array(policyOutput.data as Float32Array),
            values: new Float32Array(valueOutput.data as Float32Array),
        };
    };
}
