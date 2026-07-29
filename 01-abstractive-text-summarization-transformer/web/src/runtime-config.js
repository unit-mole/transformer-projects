export const MODEL_ID =
  'onnx-community/text_summarization-ONNX';

export const BROWSER_MODEL_LABEL =
  'Fine-tuned T5-small summarization model';

export const BROWSER_MODEL_ARCHITECTURE =
  'T5 encoder-decoder Transformer';

export const BROWSER_MODEL_DTYPE = 'fp32';

export function buildRuntimePlan() {
  return [
    {
      runtime: 'wasm',
      label: 'WASM / CPU (FP32)',
      pipelineOptions: {
        dtype: BROWSER_MODEL_DTYPE,
      },
    },
  ];
}

export function formatRuntimeError(error) {
  const raw = error instanceof Error ? error.message : String(error ?? '');
  const message = raw.trim();

  if (/out of memory|memory|allocation|oom/iu.test(message)) {
    return (
      'The browser could not allocate enough memory for the full-precision ' +
      'summarization model. Close other memory-heavy tabs, reload the Space, ' +
      'and try again.'
    );
  }

  if (/network|fetch|failed to fetch|connection|timeout/iu.test(message)) {
    return (
      'The browser could not finish downloading the ONNX model files. Check ' +
      'the connection, reload the Space, and select Load model again.'
    );
  }

  return (
    message ||
    'The browser could not initialize the full-precision summarization model.'
  );
}
