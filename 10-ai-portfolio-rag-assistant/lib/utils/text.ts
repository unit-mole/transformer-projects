const STOP_WORDS = new Set([
  "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
  "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
  "was", "were", "what", "which", "who", "with", "my", "show", "does",
]);

export function normalizeWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

export function tokenize(value: string): string[] {
  const tokens = value
    .toLowerCase()
    .replace(/[^a-z0-9+#.-]+/g, " ")
    .split(/\s+/)
    .filter(Boolean);
  return tokens.filter((token) => token.length > 1 && !STOP_WORDS.has(token));
}

export function firstSentence(value: string, maxLength = 220): string {
  const clean = normalizeWhitespace(value);
  const match = clean.match(/^(.+?[.!?])(?:\s|$)/);
  const sentence = match?.[1] ?? clean;
  return sentence.length <= maxLength
    ? sentence
    : `${sentence.slice(0, maxLength - 1).trimEnd()}…`;
}

export function safeEvidence(value: string, maxLength = 260): string {
  const clean = normalizeWhitespace(value);
  return clean.length <= maxLength
    ? clean
    : `${clean.slice(0, maxLength - 1).trimEnd()}…`;
}
