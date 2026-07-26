import { normalizeDocumentText, splitParagraphsWithOffsets } from './chunking.js';

export function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

export function locateSupportingParagraph(documentText, start, end) {
  const normalized = normalizeDocumentText(documentText);
  const paragraphs = splitParagraphsWithOffsets(normalized);
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start) {
    return { paragraph: null, paragraphs };
  }
  const paragraph = paragraphs.find((item) => start >= item.start && start < item.end)
    ?? paragraphs.find((item) => end > item.start && end <= item.end)
    ?? null;
  return { paragraph, paragraphs };
}

export function highlightEvidence(paragraph, globalStart, globalEnd) {
  if (!paragraph) {
    return '<div class="empty-evidence">No supporting paragraph could be mapped.</div>';
  }
  const localStart = globalStart - paragraph.start;
  const localEnd = globalEnd - paragraph.start;
  if (localStart < 0 || localEnd <= localStart || localEnd > paragraph.text.length) {
    return `<div class="evidence-text">${escapeHtml(paragraph.text)}</div>`;
  }
  return [
    '<div class="evidence-text">',
    escapeHtml(paragraph.text.slice(0, localStart)),
    '<mark>',
    escapeHtml(paragraph.text.slice(localStart, localEnd)),
    '</mark>',
    escapeHtml(paragraph.text.slice(localEnd)),
    '</div>',
  ].join('');
}
