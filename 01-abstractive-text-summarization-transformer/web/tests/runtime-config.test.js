import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MODEL_ID,
  buildRuntimePlan,
  formatRuntimeError,
} from '../src/runtime-config.js';

test('uses the compatible WASM q8 plan by default', () => {
  const plan = buildRuntimePlan('wasm', true);
  assert.equal(plan.length, 1);
  assert.equal(plan[0].runtime, 'wasm');
  assert.equal(plan[0].pipelineOptions.device, undefined);
  assert.deepEqual(plan[0].pipelineOptions.dtype, {
    encoder_model: 'q8',
    decoder_model_merged: 'q8',
  });
});

test('explicit WebGPU selection retains a WASM fallback', () => {
  const plan = buildRuntimePlan('webgpu', true);
  assert.deepEqual(
    plan.map((candidate) => candidate.runtime),
    ['webgpu', 'wasm'],
  );
  assert.equal(plan[0].pipelineOptions.device, 'webgpu');
  assert.deepEqual(plan[0].pipelineOptions.dtype, {
    encoder_model: 'q4',
    decoder_model_merged: 'q4',
  });
});

test('unavailable WebGPU never attempts a GPU session', () => {
  const plan = buildRuntimePlan('webgpu', false);
  assert.deepEqual(
    plan.map((candidate) => candidate.runtime),
    ['wasm'],
  );
});

test('numeric runtime failures become readable guidance', () => {
  const message = formatRuntimeError(new Error('341729896'));
  assert.match(message, /low-level numeric model-load error/u);
  assert.match(message, /WASM/u);
});

test('model id remains the browser-compatible DistilBART checkpoint', () => {
  assert.equal(MODEL_ID, 'Xenova/distilbart-cnn-12-6');
});
