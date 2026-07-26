export function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

export function clamp(value, min, max) {
  return Math.min(Math.max(Number(value), min), max);
}

export function formatMilliseconds(value) {
  if (!Number.isFinite(value)) return '—';
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}

export function detectWebGpu() {
  return Boolean(globalThis.navigator?.gpu);
}

export function safeModelId(value) {
  const candidate = value.trim();
  if (!candidate) return '';
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(candidate)) {
    throw new Error('Enter a public Hugging Face model ID in the form username/repository-name.');
  }
  return candidate;
}

export function setHidden(element, hidden) {
  element.hidden = hidden;
}
