import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MODEL_ID,
  BROWSER_MODEL_LABEL,
  BROWSER_MODEL_ARCHITECTURE,
  BROWSER_MODEL_DTYPE,
  buildRuntimePlan,
  formatRuntimeError,
} from '../src/runtime-config.js';

test('uses the official T5-small summarization ONNX model', () => {
  assert.equal(
    MODEL_ID,
    'onnx-community/text_summarization-ONNX',
  );
  assert.match(BROWSER_MODEL_LABEL, /T5-small/u);
  assert.match(BROWSER_MODEL_ARCHITECTURE, /encoder-decoder Transformer/u);
});

test('uses a single full-precision WASM runtime', () => {
  const plan = buildRuntimePlan('wasm');

  assert.equal(plan.length, 1);
  assert.equal(plan[0].runtime, 'wasm');
  assert.equal(plan[0].pipelineOptions.device, undefined);
  assert.equal(plan[0].pipelineOptions.dtype, 'fp32');
  assert.equal(BROWSER_MODEL_DTYPE, 'fp32');
});

test('does not request any quantized browser dtype', () => {
  const serialized = JSON.stringify(buildRuntimePlan());
  for (const forbidden of ['q8', 'int8', 'uint8', 'q4', 'q4f16', 'bnb4']) {
    assert.equal(serialized.includes(forbidden), false);
  }
});

test('formats memory errors with recovery guidance', () => {
  const message = formatRuntimeError(
    new Error('WASM out of memory'),
  );
  assert.match(message, /Close other memory-heavy tabs/u);
});

test('formats download errors with recovery guidance', () => {
  const message = formatRuntimeError(
    new Error('Failed to fetch model file'),
  );
  assert.match(message, /could not finish downloading/u);
});
