import test from 'node:test';
import assert from 'node:assert/strict';

import {
  MODEL_ID,
  MODEL_REVISION,
  buildRuntimePlan,
  formatRuntimeError,
} from '../src/runtime-config.js';

test('uses explicit int8 then uint8 on WASM', () => {
  const plan = buildRuntimePlan('wasm', true);

  assert.deepEqual(
    plan.map((candidate) => candidate.label),
    [
      'WASM / CPU (int8)',
      'WASM / CPU (uint8 fallback)',
    ],
  );

  assert.deepEqual(plan[0].pipelineOptions.dtype, {
    encoder_model: 'int8',
    decoder_model_merged: 'int8',
  });

  assert.deepEqual(plan[1].pipelineOptions.dtype, {
    encoder_model: 'uint8',
    decoder_model_merged: 'uint8',
  });

  assert.equal(plan[0].pipelineOptions.device, undefined);
});

test('never uses the q8 alias for the compatibility path', () => {
  const plan = buildRuntimePlan('wasm', true);
  const serialized = JSON.stringify(plan);
  assert.equal(serialized.includes('"q8"'), false);
  assert.equal(serialized.includes('quantized'), false);
});

test('explicit WebGPU keeps int8 and uint8 fallbacks', () => {
  const plan = buildRuntimePlan('webgpu', true);
  assert.deepEqual(
    plan.map((candidate) => candidate.runtime),
    ['webgpu', 'wasm', 'wasm'],
  );
  assert.deepEqual(plan[0].pipelineOptions.dtype, {
    encoder_model: 'q4',
    decoder_model_merged: 'q4',
  });
});

test('unavailable WebGPU starts directly with int8 WASM', () => {
  const plan = buildRuntimePlan('webgpu', false);
  assert.equal(plan[0].label, 'WASM / CPU (int8)');
});

test('pins the model revision containing explicit int8 files', () => {
  assert.equal(MODEL_ID, 'Xenova/distilbart-cnn-12-6');
  assert.equal(
    MODEL_REVISION,
    'a6c58857723a89bde6162f7cd64a80fd644711f6',
  );
});

test('explains the legacy q8 MatMulNBits failure', () => {
  const message = formatRuntimeError(
    new Error(
      "qdq_actions.cc:137 MatMulNBits Missing required scale: " +
      "decoder_model_merged_quantized.onnx",
    ),
  );
  assert.match(message, /legacy q8\/quantized decoder graph/u);
  assert.match(message, /int8 or uint8/u);
});
