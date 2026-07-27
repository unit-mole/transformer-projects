import { access, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const args = process.argv.slice(2);
const targetIndex = args.indexOf('--target');
const target = targetIndex >= 0 ? args[targetIndex + 1] : 'web';

const projectRoot = resolve(import.meta.dirname, '..');
const repositoryRoot = resolve(projectRoot, '..');
const roots = {
  web: resolve(projectRoot, 'web'),
  docs: resolve(repositoryRoot, 'docs', '09-vision-language-image-text-retrieval-clip'),
};

if (!Object.hasOwn(roots, target)) {
  throw new Error(`Unknown target "${target}". Use --target web or --target docs.`);
}

const root = roots[target];
const required = [
  'index.html', 'style.css', 'app.js', 'clip_preprocessing.js', 'clip_inference.js',
  'retrieval.js', 'zero_shot.js', 'metadata.json', 'zero_shot_labels.json',
  'data/image_gallery.json', 'data/image_embeddings.json', 'data/captions.json',
  'data/retrieval_eval_queries.json', 'model/README.md', 'model/model_manifest.json',
];

for (const file of required) await access(resolve(root, file));

const gallery = JSON.parse(await readFile(resolve(root, 'data/image_gallery.json'), 'utf8'));
if (!Array.isArray(gallery.images) || gallery.images.length < 1) {
  throw new Error(`${target}: gallery is empty.`);
}

for (const item of gallery.images) {
  if (!item.image_path?.startsWith('./')) {
    throw new Error(`${target}: image_path must begin with ./ for ${item.image_id}.`);
  }
  await access(resolve(root, item.image_path.replace(/^\.\//, '')));
}

const html = await readFile(resolve(root, 'index.html'), 'utf8');
if (!html.includes('src="./app.js"') || !html.includes('href="./style.css"')) {
  throw new Error(`${target}: index.html must use relative app.js and style.css references.`);
}

console.log(`Validated ${target}: ${required.length} static assets and ${gallery.images.length} gallery images.`);
