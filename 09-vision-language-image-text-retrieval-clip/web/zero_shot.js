import { cosineSimilarity } from './retrieval.js';
import { buildLabelPrompts } from './clip_preprocessing.js';

export function softmax(values, temperature = 0.01) {
  if (!values.length) return [];
  const scaled = values.map((value) => value / temperature);
  const max = Math.max(...scaled);
  const exp = scaled.map((value) => Math.exp(value - max));
  const total = exp.reduce((sum, value) => sum + value, 0);
  return exp.map((value) => value / total);
}

export async function classifyZeroShot(runtime, imageSource, labels, template = 'a photo of a {label}') {
  const prompts = buildLabelPrompts(labels, template);
  const [imageEmbedding, textEmbeddings] = await Promise.all([
    runtime.encodeImage(imageSource),
    runtime.encodeText(prompts),
  ]);
  const similarities = textEmbeddings.map((embedding) => cosineSimilarity(imageEmbedding, embedding));
  const probabilities = softmax(similarities);
  return labels
    .map((label, index) => ({ label, prompt: prompts[index], score: similarities[index], probability: probabilities[index] }))
    .sort((a, b) => b.score - a.score)
    .map((item, index) => ({ ...item, rank: index + 1 }));
}
