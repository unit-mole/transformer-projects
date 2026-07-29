export const MODEL_ID = 'Xenova/distilbart-cnn-12-6';

// Pin the repository revision that added the explicit v3-compatible int8,
// uint8, q4, and q4f16 ONNX files.
export const MODEL_REVISION =
  'a6c58857723a89bde6162f7cd64a80fd644711f6';

const INT8_DTYPES = Object.freeze({
  encoder_model: 'int8',
  decoder_model_merged: 'int8',
});

const UINT8_DTYPES = Object.freeze({
  encoder_model: 'uint8',
  decoder_model_merged: 'uint8',
});

const WEBGPU_DTYPES = Object.freeze({
  encoder_model: 'q4',
  decoder_model_merged: 'q4',
});

function cloneDtypes(value) {
  return { ...value };
}

function wasmInt8Candidate() {
  return {
    runtime: 'wasm',
    label: 'WASM / CPU (int8)',
    pipelineOptions: {
      // Omitting `device` selects the browser WASM backend.
      // `int8` intentionally selects *_int8.onnx instead of the older
      // *_quantized.onnx files selected by `q8`.
      dtype: cloneDtypes(INT8_DTYPES),
      revision: MODEL_REVISION,
    },
  };
}

function wasmUint8Candidate() {
  return {
    runtime: 'wasm',
    label: 'WASM / CPU (uint8 fallback)',
    pipelineOptions: {
      dtype: cloneDtypes(UINT8_DTYPES),
      revision: MODEL_REVISION,
    },
  };
}

function webgpuCandidate() {
  return {
    runtime: 'webgpu',
    label: 'WebGPU (q4, experimental)',
    pipelineOptions: {
      device: 'webgpu',
      dtype: cloneDtypes(WEBGPU_DTYPES),
      revision: MODEL_REVISION,
    },
  };
}

export function buildRuntimePlan(preference, webgpuAvailable) {
  const compatibleWasmPlan = [
    wasmInt8Candidate(),
    wasmUint8Candidate(),
  ];

  if (preference === 'webgpu' && webgpuAvailable) {
    return [
      webgpuCandidate(),
      ...compatibleWasmPlan,
    ];
  }

  return compatibleWasmPlan;
}

export function formatRuntimeError(error) {
  const raw = error instanceof Error ? error.message : String(error ?? '');
  const message = raw.trim();

  if (
    /qdq_actions|MatMulNBits|missing required scale|decoder_model_merged_quantized/iu.test(
      message,
    )
  ) {
    return (
      'ONNX Runtime rejected the legacy q8/quantized decoder graph. ' +
      'The application will retry with the explicit int8 or uint8 model files.'
    );
  }

  if (/^\d+$/u.test(message)) {
    return (
      'The browser runtime returned a low-level numeric model-load error. ' +
      'The application will retry with its compatibility WASM model.'
    );
  }

  if (/out of memory|memory|allocation|oom/iu.test(message)) {
    return (
      'The browser could not allocate enough memory for the model. Close other ' +
      'GPU-heavy tabs, reload the Space, and use WASM / CPU.'
    );
  }

  if (/webgpu|gpu adapter|device lost|shader|wgsl/iu.test(message)) {
    return (
      'The WebGPU backend could not initialize this encoder-decoder model. ' +
      'The application will retry with WASM / CPU.'
    );
  }

  return message || 'The browser could not initialize the summarization model.';
}
