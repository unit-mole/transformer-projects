function l2Norm(vector) {
  let sum = 0;
  for (const value of vector) sum += value * value;
  return Math.sqrt(sum);
}

export function normalizeVector(vector) {
  const norm = l2Norm(vector);
  if (!Number.isFinite(norm) || norm === 0) {
    throw new Error('Cannot normalize an empty or invalid embedding.');
  }
  return Array.from(vector, (value) => value / norm);
}

export function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) {
    throw new Error('Embeddings must be non-empty vectors with equal dimensions.');
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}

export function rankByEmbedding(queryEmbedding, gallery, embeddingMap, topK = 6) {
  return gallery
    .map((image) => {
      const vector = embeddingMap.get(image.image_id);
      if (!vector) return null;
      return { ...image, score: cosineSimilarity(queryEmbedding, vector), retrieval_mode: 'CLIP' };
    })
    .filter(Boolean)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((item, index) => ({ ...item, rank: index + 1 }));
}

function tokenize(text) {
  return String(text).toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

export function captionBaseline(query, gallery, topK = 6) {
  const documents = gallery.map((item) => tokenize(`${item.caption} ${item.category} ${item.tags.join(' ')}`));
  const df = new Map();
  for (const tokens of documents) {
    for (const token of new Set(tokens)) df.set(token, (df.get(token) ?? 0) + 1);
  }
  const qTokens = tokenize(query);
  const n = documents.length;
  const scored = gallery.map((item, index) => {
    const docTokens = documents[index];
    const counts = new Map();
    for (const token of docTokens) counts.set(token, (counts.get(token) ?? 0) + 1);
    let score = 0;
    for (const token of qTokens) {
      const tf = (counts.get(token) ?? 0) / Math.max(docTokens.length, 1);
      const idf = Math.log((n + 1) / ((df.get(token) ?? 0) + 1)) + 1;
      score += tf * idf;
    }
    return { ...item, score, retrieval_mode: 'Caption TF-IDF baseline' };
  });
  return scored
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map((item, index) => ({ ...item, rank: index + 1 }));
}

export function parseBundledEmbeddings(payload, gallery) {
  if (!payload || payload.generated !== true || !Array.isArray(payload.vectors)) return null;
  if (payload.vectors.length !== gallery.length) return null;
  const map = new Map();
  for (const record of payload.vectors) {
    if (!record.image_id || !Array.isArray(record.embedding)) return null;
    map.set(record.image_id, normalizeVector(record.embedding));
  }
  return map.size === gallery.length ? map : null;
}
