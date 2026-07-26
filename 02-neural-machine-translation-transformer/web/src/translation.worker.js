import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';

env.allowLocalModels = false;
env.useBrowserCache = true;

const MODELS = {
  'en-hi': 'Xenova/opus-mt-en-hi',
  'hi-en': 'Xenova/opus-mt-hi-en',
};
const cache = new Map();

function progressMessage(direction, item) {
  const raw = Number(item?.progress);
  const progress = Number.isFinite(raw) ? Math.max(0, Math.min(100, raw)) : null;
  self.postMessage({
    type: 'progress',
    direction,
    status: item?.status ?? 'loading',
    file: item?.file ?? '',
    progress,
  });
}

async function createPipeline(direction) {
  const model = MODELS[direction];
  if (!model) throw new Error(`Unsupported direction: ${direction}`);

  self.postMessage({ type: 'model-status', direction, message: `Loading ${model} with q4 quantization…` });
  try {
    return await pipeline('translation', model, {
      dtype: 'q4',
      progress_callback: (item) => progressMessage(direction, item),
    });
  } catch (q4Error) {
    self.postMessage({ type: 'model-status', direction, message: 'q4 loading failed; retrying with q8 for compatibility…' });
    return pipeline('translation', model, {
      dtype: 'q8',
      progress_callback: (item) => progressMessage(direction, item),
    });
  }
}

async function getPipeline(direction) {
  if (!cache.has(direction)) cache.set(direction, createPipeline(direction));
  try {
    return await cache.get(direction);
  } catch (error) {
    cache.delete(direction);
    throw error;
  }
}

self.addEventListener('message', async (event) => {
  const request = event.data ?? {};
  if (request.type !== 'translate') return;

  try {
    const translator = await getPipeline(request.direction);
    const sourceTokens = translator.tokenizer.tokenize(request.text, { add_special_tokens: true });
    const started = performance.now();
    self.postMessage({ type: 'inference-status', id: request.id, message: 'Model ready. Running encoder-decoder generation…' });

    const result = await translator(request.text, {
      num_beams: Number(request.numBeams ?? 4),
      max_new_tokens: Number(request.maxNewTokens ?? 128),
      do_sample: false,
      early_stopping: true,
      no_repeat_ngram_size: 3,
    });
    const translatedText = result?.[0]?.translation_text ?? '';
    const targetTokens = translator.tokenizer.tokenize(translatedText, { add_special_tokens: true });
    const latencyMs = performance.now() - started;

    self.postMessage({
      type: 'translation-result',
      id: request.id,
      direction: request.direction,
      model: MODELS[request.direction],
      translatedText,
      sourceTokens: sourceTokens.slice(0, 80),
      sourceTokenCount: sourceTokens.length,
      targetTokens: targetTokens.slice(0, 80),
      targetTokenCount: targetTokens.length,
      latencyMs,
    });
  } catch (error) {
    self.postMessage({
      type: 'translation-error',
      id: request.id,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
