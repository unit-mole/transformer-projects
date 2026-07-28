export const SUPPORTED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp', 'image/bmp']);

export function validateImageFile(file, maxMb = 10) {
  if (!(file instanceof Blob)) throw new Error('A valid image file is required.');
  if (file.type && !SUPPORTED_TYPES.has(file.type)) throw new Error('Unsupported image type. Use JPEG, PNG, WebP or BMP.');
  if (file.size > maxMb * 1024 * 1024) throw new Error(`Image exceeds ${maxMb} MB.`);
  return true;
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

export function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('The browser could not decode this image.'));
    image.src = url;
  });
}

export async function fileFromUrl(url, name = 'sample.svg') {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load sample image (${response.status}).`);
  const blob = await response.blob();
  return new File([blob], name, { type: blob.type || 'image/svg+xml' });
}

export function canvasWithMaskedPatch(image, row, col, gridSize, mask = '#7f7f7f') {
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  const context = canvas.getContext('2d', { willReadFrequently: false });
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  const patchWidth = canvas.width / gridSize;
  const patchHeight = canvas.height / gridSize;
  context.fillStyle = mask;
  context.fillRect(col * patchWidth, row * patchHeight, patchWidth, patchHeight);
  return canvas;
}
