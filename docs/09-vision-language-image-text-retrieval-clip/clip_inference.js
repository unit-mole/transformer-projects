import {
  AutoProcessor,
  AutoTokenizer,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  RawImage,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0';
import { normalizeVector } from './retrieval.js';

const MODEL_ID = 'Xenova/clip-vit-base-patch32';
const DTYPE = 'q8';
const DEVICE = 'wasm';
const CACHE_KEY = 'clip-gallery-embeddings-v1';

env.allowRemoteModels = true;
env.allowLocalModels = false;
env.useBrowserCache = true;

function rowsFromTensor(tensor) {
  const dims = tensor.dims;
  const rows = dims.length === 1 ? 1 : dims[0];
  const width = tensor.data.length / rows;
  const output = [];
  for (let row = 0; row < rows; row += 1) {
    output.push(normalizeVector(Array.from(tensor.data.slice(row * width, (row + 1) * width))));
  }
  return output;
}

export class ClipRuntime {
  constructor({ progressCallback = () => {} } = {}) {
    this.progressCallback = progressCallback;
    this.tokenizer = null;
    this.processor = null;
    this.textModel = null;
    this.visionModel = null;
  }

  handleProgress = (event) => {
    this.progressCallback(event);
  };

  async loadText() {
    if (this.tokenizer && this.textModel) return;
    this.progressCallback({ status: 'initiate', file: 'CLIP text encoder' });
    const [tokenizer, model] = await Promise.all([
      AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: this.handleProgress }),
      CLIPTextModelWithProjection.from_pretrained(MODEL_ID, {
        dtype: DTYPE,
        device: DEVICE,
        progress_callback: this.handleProgress,
      }),
    ]);
    this.tokenizer = tokenizer;
    this.textModel = model;
    this.progressCallback({ status: 'ready', file: 'CLIP text encoder' });
  }

  async loadVision() {
    if (this.processor && this.visionModel) return;
    this.progressCallback({ status: 'initiate', file: 'CLIP vision encoder' });
    const [processor, model] = await Promise.all([
      AutoProcessor.from_pretrained(MODEL_ID, { progress_callback: this.handleProgress }),
      CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, {
        dtype: DTYPE,
        device: DEVICE,
        progress_callback: this.handleProgress,
      }),
    ]);
    this.processor = processor;
    this.visionModel = model;
    this.progressCallback({ status: 'ready', file: 'CLIP vision encoder' });
  }

  async encodeText(texts) {
    await this.loadText();
    const list = Array.isArray(texts) ? texts : [texts];
    const inputs = this.tokenizer(list, { padding: true, truncation: true });
    const { text_embeds } = await this.textModel(inputs);
    return rowsFromTensor(text_embeds);
  }

  async encodeImage(source) {
    await this.loadVision();
    const image = await RawImage.read(source);
    const inputs = await this.processor(image);
    const { image_embeds } = await this.visionModel(inputs);
    return rowsFromTensor(image_embeds)[0];
  }

  async encodeGallery(gallery, onItem = () => {}) {
    const map = new Map();
    for (let index = 0; index < gallery.length; index += 1) {
      const item = gallery[index];
      const embedding = await this.encodeImage(item.image_path);
      map.set(item.image_id, embedding);
      onItem({ index: index + 1, total: gallery.length, item });
    }
    return map;
  }

  static loadCachedGallery(gallery, modelId = MODEL_ID) {
    try {
      const payload = JSON.parse(localStorage.getItem(CACHE_KEY) ?? 'null');
      if (!payload || payload.model_id !== modelId || payload.version !== 1) return null;
      if (!Array.isArray(payload.vectors) || payload.vectors.length !== gallery.length) return null;
      const expectedIds = new Set(gallery.map((item) => item.image_id));
      const map = new Map();
      for (const row of payload.vectors) {
        if (!expectedIds.has(row.image_id) || !Array.isArray(row.embedding)) return null;
        map.set(row.image_id, normalizeVector(row.embedding));
      }
      return map.size === gallery.length ? map : null;
    } catch {
      return null;
    }
  }

  static saveCachedGallery(map, modelId = MODEL_ID) {
    try {
      const payload = {
        version: 1,
        model_id: modelId,
        created_at: new Date().toISOString(),
        vectors: [...map.entries()].map(([image_id, embedding]) => ({ image_id, embedding })),
      };
      localStorage.setItem(CACHE_KEY, JSON.stringify(payload));
      return true;
    } catch {
      return false;
    }
  }
}

export const CLIP_CONFIG = { modelId: MODEL_ID, dtype: DTYPE, device: DEVICE, cacheKey: CACHE_KEY };
