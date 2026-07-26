/** Pure search, filtering, scoring, and safe rendering helpers. */

export function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let index = 0; index < a.length; index += 1) {
    dot += a[index] * b[index];
    normA += a[index] * a[index];
    normB += b[index] * b[index];
  }
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator > 0 ? dot / denominator : 0;
}

export function rankSemantic(queryVector, chunks, vectors, options = {}) {
  const { topK = 5, category = "", documentType = "" } = options;
  return chunks
    .map((chunk, index) => ({ chunk, vector: vectors[index], index }))
    .filter(({ chunk }) => !category || chunk.project_category === category)
    .filter(({ chunk }) => !documentType || chunk.document_type === documentType)
    .map(({ chunk, vector, index }) => ({
      ...chunk,
      source_index: index,
      similarity_score: cosineSimilarity(queryVector, vector),
    }))
    .sort((left, right) => right.similarity_score - left.similarity_score)
    .slice(0, topK)
    .map((item, index) => ({ ...item, rank: index + 1 }));
}

export function keywordSearch(query, chunks, options = {}) {
  const { topK = 5, category = "", documentType = "" } = options;
  const terms = query.toLowerCase().match(/[a-z0-9][a-z0-9+.#@/-]*/g) || [];
  if (!terms.length) return [];
  return chunks
    .filter((chunk) => !category || chunk.project_category === category)
    .filter((chunk) => !documentType || chunk.document_type === documentType)
    .map((chunk) => {
      const searchable = `${chunk.project_name} ${chunk.section_title} ${chunk.text} ${(chunk.tags || []).join(" ")}`.toLowerCase();
      const score = terms.reduce((sum, term) => sum + (searchable.includes(term) ? 1 : 0), 0) / terms.length;
      return { ...chunk, similarity_score: score };
    })
    .filter((item) => item.similarity_score > 0)
    .sort((left, right) => right.similarity_score - left.similarity_score)
    .slice(0, topK)
    .map((item, index) => ({ ...item, rank: index + 1 }));
}

export function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function highlightLexicalTerms(text, query) {
  const safe = escapeHtml(text);
  const terms = [...new Set((query.toLowerCase().match(/[a-z0-9][a-z0-9+.#@/-]{2,}/g) || []))]
    .sort((a, b) => b.length - a.length)
    .slice(0, 8);
  if (!terms.length) return safe;
  const escapedTerms = terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(`(${escapedTerms.join("|")})`, "gi");
  return safe.replace(pattern, "<mark>$1</mark>");
}

export function formatDocumentType(value) {
  return String(value || "document").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
