export const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073];
export const CLIP_STD = [0.26862954, 0.26130258, 0.27577711];

export function cleanText(value, { fieldName = 'Text', maxLength = 500 } = {}) {
  if (typeof value !== 'string') {
    throw new TypeError(`${fieldName} must be text.`);
  }
  const cleaned = value.replace(/\s+/g, ' ').trim();
  if (!cleaned) {
    throw new Error(`${fieldName} cannot be empty.`);
  }
  if (cleaned.length > maxLength) {
    throw new Error(`${fieldName} must be ${maxLength} characters or fewer.`);
  }
  return cleaned;
}

export function parseCandidateLabels(value, { maxLabels = 24 } = {}) {
  const labels = String(value ?? '')
    .split(/[,\n;]/)
    .map((label) => label.replace(/\s+/g, ' ').trim())
    .filter(Boolean);
  const unique = [...new Set(labels.map((label) => label.toLowerCase()))];
  if (unique.length < 2) {
    throw new Error('Enter at least two candidate labels separated by commas.');
  }
  if (unique.length > maxLabels) {
    throw new Error(`Use ${maxLabels} candidate labels or fewer.`);
  }
  return unique;
}

export function buildLabelPrompts(labels, template = 'a photo of a {label}') {
  return labels.map((label) => template.replace('{label}', label));
}

export function validateImageFile(file, { maxBytes = 10 * 1024 * 1024 } = {}) {
  if (!(file instanceof File)) {
    throw new Error('Select an image file first.');
  }
  const accepted = new Set(['image/png', 'image/jpeg', 'image/webp']);
  if (!accepted.has(file.type)) {
    throw new Error('Use a PNG, JPEG, or WebP image.');
  }
  if (file.size > maxBytes) {
    throw new Error('The image must be 10 MB or smaller.');
  }
  return file;
}
