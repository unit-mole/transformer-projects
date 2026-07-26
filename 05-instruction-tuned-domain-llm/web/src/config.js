export const APP_CONFIG = Object.freeze({
  title: 'ML & Data Science Instruction-Tuned Assistant',
  projectNumber: '05',
  baseModel: {
    id: 'Xenova/flan-t5-small',
    label: 'Browser-compatible base FLAN-T5-small',
    claim: 'Base model demonstration — not trained by this project',
  },
  defaultCustomModelId: '',
  githubUrl: 'https://github.com/YOUR_GITHUB_USERNAME/transformer-projects/tree/main/05-instruction-tuned-domain-llm',
  modelCardUrl: 'https://github.com/YOUR_GITHUB_USERNAME/transformer-projects/blob/main/05-instruction-tuned-domain-llm/MODEL_CARD.md',
  datasetCardUrl: 'https://github.com/YOUR_GITHUB_USERNAME/transformer-projects/blob/main/05-instruction-tuned-domain-llm/DATASET_CARD.md',
  evaluationUrl: 'https://github.com/YOUR_GITHUB_USERNAME/transformer-projects/tree/main/05-instruction-tuned-domain-llm/outputs',
  maxInputCharacters: 2400,
  maxContextCharacters: 1400,
  responsibleUse:
    'Educational portfolio demo only. Responses may be incomplete, incorrect, outdated, biased, or hallucinated. Do not use for legal, medical, financial, immigration, safety-critical, official, or autonomous decisions, and do not enter private or confidential data.',
});

export function runtimeVariables() {
  return globalThis.window?.huggingface?.variables ?? {};
}
