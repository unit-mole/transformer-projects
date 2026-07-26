const DEVANAGARI = /[\u0900-\u097F]/u;
const LATIN = /[A-Za-z]/u;

export function detectLanguage(text) {
  const value = String(text ?? '').normalize('NFKC').trim();
  let devanagari = 0;
  let latin = 0;

  for (const char of value) {
    if (DEVANAGARI.test(char)) devanagari += 1;
    else if (LATIN.test(char)) latin += 1;
  }

  const letters = devanagari + latin;
  if (letters < 2) {
    return { language: 'uncertain', devanagari, latin, letters, confidence: 0 };
  }

  const devanagariRatio = devanagari / letters;
  const latinRatio = latin / letters;
  if (devanagari >= 2 && latin >= 2 && devanagariRatio >= 0.2 && latinRatio >= 0.2) {
    return { language: 'mixed', devanagari, latin, letters, confidence: Math.max(devanagariRatio, latinRatio) };
  }
  if (devanagariRatio >= 0.65) {
    return { language: 'hindi', devanagari, latin, letters, confidence: devanagariRatio };
  }
  if (latinRatio >= 0.65) {
    return { language: 'english', devanagari, latin, letters, confidence: latinRatio };
  }
  return { language: 'uncertain', devanagari, latin, letters, confidence: Math.max(devanagariRatio, latinRatio) };
}

export function resolveDirection(text, selected = 'auto') {
  const detection = detectLanguage(text);
  if (selected === 'en-hi' || selected === 'hi-en') {
    return { direction: selected, detection };
  }
  if (detection.language === 'english') return { direction: 'en-hi', detection };
  if (detection.language === 'hindi') return { direction: 'hi-en', detection };
  return { direction: null, detection };
}

export function directionLabel(direction) {
  return direction === 'en-hi' ? 'English → Hindi' : direction === 'hi-en' ? 'Hindi → English' : '—';
}
