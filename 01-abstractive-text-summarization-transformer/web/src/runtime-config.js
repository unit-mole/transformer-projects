export const MODEL_ID = 'Xenova/distilbart-cnn-12-6';

const WASM_DTYPES = Object.freeze({
  encoder_model: 'q8',
  decoder_model_merged: 'q8',
});

const WEBGPU_DTYPES = Object.freeze({
  encoder_model: 'q4',
  decoder_model_merged: 'q4',
});

function cloneDtypes(value) {
  return { ...value };
}

export function buildRuntimePlan(preference, webgpuAvailable) {
  const wasm = {
    runtime: 'wasm',
    label: 'WASM / CPU (q8)',
    pipelineOptions: {
      // Omitting `device` selects the browser's reliable WASM backend.
      dtype: cloneDtypes(WASM_DTYPES),
    },
  };

  const webgpu = {
    runtime: 'webgpu',
    label: 'WebGPU (q4)',
    pipelineOptions: {
      device: 'webgpu',
      dtype: cloneDtypes(WEBGPU_DTYPES),
    },
  };

  if (preference === 'webgpu' && webgpuAvailable) {
    // WebGPU is attempted only when the visitor explicitly selects it.
    // A clean WASM/q8 attempt follows if the experimental backend fails.
    return [webgpu, wasm];
  }

  // "wasm", "auto", missing, and unavailable-WebGPU cases all use the
  // compatibility-first path without first allocating a WebGPU session.
  return [wasm];
}

export function formatRuntimeError(error) {
  const raw = error instanceof Error ? error.message : String(error ?? '');
  const message = raw.trim();

  if (/^\d+$/u.test(message)) {
    return (
      'The browser runtime returned a low-level numeric model-load error. ' +
      'This commonly occurs when an experimental WebGPU session cannot create ' +
      'the model. Retry with the recommended WASM / CPU runtime.'
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
      'Use the recommended WASM / CPU runtime.'
    );
  }

  return message || 'The browser could not initialize the summarization model.';
}
