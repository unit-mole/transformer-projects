import { detectLanguage } from './language-detection.js';

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function confidenceProxy(source, translated, direction) {
  const input = String(source ?? '').trim();
  const output = String(translated ?? '').trim();
  if (!input || !output) return { score: 0, label: 'Unavailable', reasons: ['Missing input or output'] };

  let score = 0.84;
  const reasons = [];
  const ratio = output.length / Math.max(input.length, 1);

  if (output.toLocaleLowerCase() === input.toLocaleLowerCase()) {
    score -= 0.35;
    reasons.push('Output is unchanged from the source');
  }
  if (ratio < 0.25 || ratio > 4) {
    score -= 0.2;
    reasons.push('Unusual source-to-target length ratio');
  } else if (ratio < 0.45 || ratio > 2.5) {
    score -= 0.09;
    reasons.push('Moderately unusual output length');
  }
  if (/\b(.{2,20})(?:\s+\1){2,}\b/iu.test(output)) {
    score -= 0.16;
    reasons.push('Repeated phrase pattern detected');
  }
  if (output.includes('<unk>') || output.includes('[UNK]')) {
    score -= 0.22;
    reasons.push('Unknown token marker detected');
  }

  const target = detectLanguage(output);
  const expected = direction === 'en-hi' ? 'hindi' : 'english';
  if (target.language === expected) {
    score += 0.05;
    reasons.push('Target script matches the selected direction');
  } else if (target.language === 'mixed') {
    score -= 0.08;
    reasons.push('Mixed target scripts detected');
  } else if (target.language !== 'uncertain') {
    score -= 0.2;
    reasons.push('Target script does not match the selected direction');
  }

  score = clamp(score, 0.05, 0.95);
  const label = score >= 0.8 ? 'Higher' : score >= 0.6 ? 'Moderate' : 'Lower';
  return { score, label, reasons };
}
