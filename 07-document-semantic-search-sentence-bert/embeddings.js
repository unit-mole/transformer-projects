/** Transformers.js model loading and embedding generation. */

const TRANSFORMERS_CDN = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
export const BROWSER_MODEL = "Xenova/all-MiniLM-L6-v2";

let extractorPromise = null;

export async function getExtractor(progressCallback) {
  if (!extractorPromise) {
    extractorPromise = (async () => {
      const { env, pipeline } = await import(TRANSFORMERS_CDN);
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      return pipeline("feature-extraction", BROWSER_MODEL, {
        progress_callback: progressCallback,
      });
    })();
  }
  return extractorPromise;
}

export async function embedTexts(texts, progressCallback) {
  const values = Array.isArray(texts) ? texts : [texts];
  if (!values.length) return [];
  const extractor = await getExtractor(progressCallback);
  const output = await extractor(values, { pooling: "mean", normalize: true });
  const nested = output.tolist();
  return Array.isArray(nested[0]) ? nested : [nested];
}
