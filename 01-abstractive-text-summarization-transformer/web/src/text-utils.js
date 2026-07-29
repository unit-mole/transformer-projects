export function normalizeText(value) {
  return String(value ?? '')
    .normalize('NFKC')
    .replace(/\r\n?/g, '\n')
    .replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

export function countWords(value) {
  const text = normalizeText(value);
  return text ? text.split(/\s+/u).length : 0;
}

export function countCharacters(value) {
  return [...normalizeText(value)].length;
}

export function compressionRatio(source, summary) {
  const sourceWords = countWords(source);
  if (sourceWords === 0) return 0;
  return countWords(summary) / sourceWords;
}

export function formatCompressionRatio(source, summary) {
  const ratio = compressionRatio(source, summary);
  return ratio === 0 ? '—' : `${(ratio * 100).toFixed(1)}%`;
}

export function formatDuration(milliseconds) {
  const value = Number(milliseconds);
  if (!Number.isFinite(value) || value < 0) return '—';
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}

export function splitIntoSentences(value) {
  const text = normalizeText(value);
  if (!text) return [];

  const matches = text.match(/[^.!?\n]+(?:[.!?]+|$)/gu);
  return (matches ?? [text]).map((sentence) => sentence.trim()).filter(Boolean);
}

export function validateGenerationSettings(settings) {
  const minNewTokens = Number(settings.minNewTokens);
  const maxNewTokens = Number(settings.maxNewTokens);
  const numBeams = Number(settings.numBeams);
  const lengthPenalty = Number(settings.lengthPenalty);
  const noRepeatNgramSize = Number(settings.noRepeatNgramSize);

  if (!Number.isInteger(minNewTokens) || minNewTokens < 1) {
    throw new Error('Minimum new tokens must be a positive integer.');
  }
  if (!Number.isInteger(maxNewTokens) || maxNewTokens <= minNewTokens) {
    throw new Error('Maximum new tokens must be greater than minimum new tokens.');
  }
  if (!Number.isInteger(numBeams) || numBeams < 1 || numBeams > 8) {
    throw new Error('Beam count must be an integer between 1 and 8.');
  }
  if (!Number.isFinite(lengthPenalty) || lengthPenalty < 0.1 || lengthPenalty > 5) {
    throw new Error('Length penalty must be between 0.1 and 5.0.');
  }
  if (!Number.isInteger(noRepeatNgramSize) || noRepeatNgramSize < 0 || noRepeatNgramSize > 10) {
    throw new Error('No-repeat n-gram size must be an integer between 0 and 10.');
  }

  return {
    minNewTokens,
    maxNewTokens,
    numBeams,
    lengthPenalty,
    noRepeatNgramSize,
    earlyStopping: true,
  };
}

export function buildDownloadFileName(prefix = 'distilbart-summary') {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  return `${prefix}-${stamp}.txt`;
}
