import { AutoTokenizer, env, pipeline } from '@huggingface/transformers';

env.allowLocalModels = false;

const cache = new Map();
let activeKey = null;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function progressCallback(progress) {
  post('progress', { progress });
}

async function createRuntime({ modelId, device, dtype }) {
  const key = `${modelId}|${device}|${dtype}`;
  if (cache.has(key)) {
    activeKey = key;
    return cache.get(key);
  }

  post('status', { message: `Loading tokenizer for ${modelId}…` });
  const tokenizer = await AutoTokenizer.from_pretrained(modelId, {
    progress_callback: progressCallback,
  });

  post('status', { message: `Loading ${dtype} model on ${device.toUpperCase()}…` });
  const generator = await pipeline('text2text-generation', modelId, {
    device,
    dtype,
    progress_callback: progressCallback,
  });

  const runtime = { tokenizer, generator, modelId, device, dtype };
  cache.set(key, runtime);
  activeKey = key;
  return runtime;
}

async function loadWithFallback(options) {
  try {
    return await createRuntime(options);
  } catch (firstError) {
    if (options.device === 'webgpu') {
      post('status', { message: 'WebGPU load failed; retrying with WASM and q8 weights…' });
      try {
        return await createRuntime({ ...options, device: 'wasm', dtype: 'q8' });
      } catch (fallbackError) {
        fallbackError.cause = firstError;
        throw fallbackError;
      }
    }
    throw firstError;
  }
}

self.addEventListener('message', async (event) => {
  const { id, action, payload } = event.data;
  try {
    if (action === 'load') {
      const runtime = await loadWithFallback(payload);
      post('loaded', {
        id,
        metadata: {
          modelId: runtime.modelId,
          device: runtime.device,
          dtype: runtime.dtype,
        },
      });
      return;
    }

    if (action === 'generate') {
      const runtime = activeKey ? cache.get(activeKey) : await loadWithFallback(payload.runtime);
      if (!runtime) throw new Error('No model runtime is available. Load the model first.');

      const tokens = runtime.tokenizer.tokenize(payload.prompt, { add_special_tokens: true });
      const tokenIds = runtime.tokenizer.encode(payload.prompt, { add_special_tokens: true });
      const generationOptions = {
        max_new_tokens: payload.settings.maxNewTokens,
        repetition_penalty: payload.settings.repetitionPenalty,
      };

      if (payload.settings.temperature > 0) {
        generationOptions.do_sample = true;
        generationOptions.temperature = payload.settings.temperature;
        generationOptions.top_p = payload.settings.topP;
      } else {
        generationOptions.do_sample = false;
      }

      const started = performance.now();
      const output = await runtime.generator(payload.prompt, generationOptions);
      const latencyMs = performance.now() - started;
      const generatedText = output?.[0]?.generated_text?.trim() ?? '';
      const outputTokens = runtime.tokenizer.tokenize(generatedText, { add_special_tokens: false });

      post('result', {
        id,
        result: {
          generatedText,
          latencyMs,
          sourceTokenCount: tokenIds.length,
          targetTokenCount: outputTokens.length,
          tokenPreview: tokens.slice(0, 80),
          tokenIdsPreview: tokenIds.slice(0, 80),
          modelId: runtime.modelId,
          device: runtime.device,
          dtype: runtime.dtype,
        },
      });
      return;
    }

    throw new Error(`Unknown worker action: ${action}`);
  } catch (error) {
    post('error', {
      id,
      error: {
        name: error?.name ?? 'Error',
        message: error?.message ?? String(error),
        cause: error?.cause?.message ?? null,
      },
    });
  }
});
