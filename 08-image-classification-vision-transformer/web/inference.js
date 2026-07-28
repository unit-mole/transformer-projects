import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/+esm';

env.useBrowserCache = true;
env.allowRemoteModels = true;

let classifier = null;
let activeDevice = 'wasm';

async function createPipeline(metadata, device) {
  const options = { dtype: metadata.browser_model.dtype || 'q8' };
  if (device === 'webgpu') options.device = 'webgpu';
  return pipeline(metadata.browser_model.task, metadata.browser_model.model_id, options);
}

export async function loadClassifier(metadata, onStatus = () => {}) {
  if (classifier) return { classifier, device: activeDevice };
  const canUseWebGpu = Boolean(navigator.gpu) && metadata.browser_model.preferred_device === 'webgpu';
  if (canUseWebGpu) {
    try {
      onStatus('Loading WebGPU model…');
      classifier = await createPipeline(metadata, 'webgpu');
      activeDevice = 'webgpu';
      return { classifier, device: activeDevice };
    } catch (error) {
      console.warn('WebGPU initialization failed; retrying with WASM.', error);
      classifier = null;
    }
  }
  onStatus('Loading WebAssembly model…');
  classifier = await createPipeline(metadata, 'wasm');
  activeDevice = 'wasm';
  return { classifier, device: activeDevice };
}

export async function classifyImage(imageSource, metadata, onStatus = () => {}) {
  const loaded = await loadClassifier(metadata, onStatus);
  const start = performance.now();
  const predictions = await loaded.classifier(imageSource, { topk: metadata.prediction.top_k || 5 });
  const latencyMs = performance.now() - start;
  return { predictions, latencyMs, device: loaded.device };
}

export function findClassScore(predictions, label) {
  const match = predictions.find((item) => item.label === label);
  return match ? Number(match.score) : 0;
}
