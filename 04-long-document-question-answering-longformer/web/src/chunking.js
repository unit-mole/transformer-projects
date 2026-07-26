const DEFAULT_STOP_WORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'how',
  'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was',
  'what', 'when', 'where', 'which', 'who', 'why', 'with', 'were', 'will',
]);

export function normalizeDocumentText(value) {
  return String(value ?? '')
    .replace(/\r\n?/g, '\n')
    .replace(/[\t\f\v]+/g, ' ')
    .replace(/[ ]{2,}/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

export function countWords(value) {
  const matches = normalizeDocumentText(value).match(/\S+/g);
  return matches ? matches.length : 0;
}

export function splitParagraphsWithOffsets(text) {
  const normalized = normalizeDocumentText(text);
  if (!normalized) return [];

  const paragraphs = [];
  const pattern = /\S(?:[\s\S]*?\S)?(?=\n\s*\n|$)/g;
  let match;
  while ((match = pattern.exec(normalized)) !== null) {
    const paragraphText = match[0].trim();
    if (!paragraphText) continue;
    const relativeStart = match[0].indexOf(paragraphText);
    const start = match.index + Math.max(0, relativeStart);
    paragraphs.push({
      id: paragraphs.length,
      text: paragraphText,
      start,
      end: start + paragraphText.length,
    });
  }
  return paragraphs;
}

export function chunkDocument(text, maxWords = 260, overlapWords = 60) {
  const normalized = normalizeDocumentText(text);
  if (!normalized) return [];

  const safeMaxWords = Math.max(80, Number(maxWords) || 260);
  const safeOverlap = Math.min(
    Math.max(0, Number(overlapWords) || 0),
    Math.max(0, safeMaxWords - 20),
  );
  const wordMatches = [...normalized.matchAll(/\S+/g)];
  if (!wordMatches.length) return [];

  const chunks = [];
  const step = Math.max(20, safeMaxWords - safeOverlap);
  for (let startWord = 0; startWord < wordMatches.length; startWord += step) {
    const endWordExclusive = Math.min(startWord + safeMaxWords, wordMatches.length);
    const start = wordMatches[startWord].index;
    const lastMatch = wordMatches[endWordExclusive - 1];
    const end = lastMatch.index + lastMatch[0].length;
    const chunkText = normalized.slice(start, end);
    chunks.push({
      id: chunks.length,
      text: chunkText,
      start,
      end,
      wordCount: endWordExclusive - startWord,
    });
    if (endWordExclusive >= wordMatches.length) break;
  }
  return chunks;
}

export function extractQueryTerms(question, stopWords = DEFAULT_STOP_WORDS) {
  return [...new Set(
    normalizeDocumentText(question)
      .toLowerCase()
      .match(/[a-z0-9][a-z0-9_-]*/g) ?? [],
  )].filter((term) => term.length > 1 && !stopWords.has(term));
}

export function lexicalScore(question, text) {
  const terms = extractQueryTerms(question);
  if (!terms.length) return 0;
  const lowerText = normalizeDocumentText(text).toLowerCase();
  let score = 0;
  for (const term of terms) {
    const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const occurrences = lowerText.match(new RegExp(`\\b${escaped}\\b`, 'g'))?.length ?? 0;
    score += Math.min(occurrences, 4);
  }
  const normalizedQuestion = normalizeDocumentText(question).toLowerCase();
  if (normalizedQuestion.length >= 8 && lowerText.includes(normalizedQuestion)) score += 5;
  return score / Math.max(1, terms.length);
}

export function rankChunks(question, chunks, limit = 6) {
  const safeLimit = Math.max(1, Math.min(Number(limit) || 6, chunks.length || 1));
  return chunks
    .map((chunk) => ({ ...chunk, retrievalScore: lexicalScore(question, chunk.text) }))
    .sort((a, b) => b.retrievalScore - a.retrievalScore || a.id - b.id)
    .slice(0, safeLimit);
}
