import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const required = [
  'README.md', 'index.html', 'package.json', 'vite.config.js',
  'src/main.js', 'src/model-worker.js', 'src/model-client.js',
  'src/prompt-templates.js', 'src/config.js', 'src/styles.css',
  'public/model-metadata.json', 'public/evaluation-summary.json', 'public/architecture.svg',
];

const missing = required.filter((file) => !fs.existsSync(path.join(root, file)));
if (missing.length) {
  console.error(`Missing required Static Space files:\n${missing.join('\n')}`);
  process.exit(1);
}

const readme = fs.readFileSync(path.join(root, 'README.md'), 'utf8');
for (const value of ['sdk: static', 'app_build_command: npm run build', 'app_file: dist/index.html']) {
  if (!readme.includes(value)) {
    console.error(`README.md is missing required metadata: ${value}`);
    process.exit(1);
  }
}

const metadata = JSON.parse(fs.readFileSync(path.join(root, 'public/model-metadata.json'), 'utf8'));
if (metadata.custom_model_status !== 'not_published') {
  console.error('Starter package must not claim that a custom model is already published.');
  process.exit(1);
}

console.log('Static Space configuration is valid.');
