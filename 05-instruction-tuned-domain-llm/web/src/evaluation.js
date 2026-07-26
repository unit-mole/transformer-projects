const ML_TERMS = new Set([
  'accuracy', 'algorithm', 'analytics', 'auc', 'bias', 'classification', 'clustering',
  'cnn', 'cross-validation', 'data', 'dataset', 'decision', 'feature', 'f1', 'forest',
  'learning', 'loss', 'lstm', 'metric', 'model', 'neural', 'precision', 'prediction',
  'recall', 'regression', 'regularization', 'rnn', 'training', 'transformer', 'validation',
]);

function words(text) {
  return text.toLowerCase().match(/[a-z0-9-]+/g) ?? [];
}

function jaccard(left, right) {
  const a = new Set(words(left).filter((word) => word.length > 2));
  const b = new Set(words(right).filter((word) => word.length > 2));
  if (!a.size || !b.size) return 0;
  const intersection = [...a].filter((item) => b.has(item)).length;
  const union = new Set([...a, ...b]).size;
  return intersection / union;
}

export function demoEvaluation(prompt, response) {
  const responseWords = words(response);
  const topicalTerms = responseWords.filter((word) => ML_TERMS.has(word));
  const relevance = Math.min(1, jaccard(prompt, response) * 2.5 + Math.min(topicalTerms.length / 8, 0.35));

  const lengthScore = responseWords.length >= 25 ? 1 : responseWords.length / 25;
  const adherence = Math.min(1, 0.55 * relevance + 0.45 * lengthScore);

  const overconfidencePatterns = [
    /\balways\b/i,
    /\bnever\b/i,
    /\bguarantee(?:d|s)?\b/i,
    /\b100%\b/i,
    /\bperfect(?:ly)?\b/i,
  ];
  const warnings = overconfidencePatterns
    .filter((pattern) => pattern.test(response))
    .map((pattern) => `Review absolute wording matched by ${pattern}`);

  if (responseWords.length < 12) warnings.push('Response is very short and may omit important caveats.');
  if (!topicalTerms.length) warnings.push('Low visible ML/Data Science terminology; manually verify topical relevance.');

  return {
    adherenceScore: Number(adherence.toFixed(2)),
    relevanceScore: Number(relevance.toFixed(2)),
    hallucinationRisk: warnings.length >= 2 ? 'Review required' : warnings.length === 1 ? 'Possible' : 'No simple flag',
    warnings,
    disclaimer: 'These are transparent browser heuristics for demo diagnostics, not official model-evaluation results.',
  };
}
