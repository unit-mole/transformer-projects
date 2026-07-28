export const APP_CONFIG = Object.freeze({
  projectName: '04-long-document-question-answering-longformer',

  // Transformer executed directly in the browser.
  browserModelId: 'Xenova/distilbert-base-cased-distilled-squad',

  // Fine-tuned Longformer produced and evaluated by the Python project.
  pythonModelId: 'anmol-unitmole/longformer-qasper-document-qa',

  browserArchitecture:
    'DistilBERT extractive QA browser deployment baseline',

  pythonArchitecture:
    'QASPER-fine-tuned Longformer extractive QA implementation',

  defaultChunkWords: 260,
  defaultOverlapWords: 60,
  defaultCandidateChunks: 6,
  maximumDocumentCharacters: 1_000_000,

  githubUrl:
    'https://github.com/unit-mole/transformer-projects/tree/main/04-long-document-question-answering-longformer',

  // The interface currently expects this property name. It now links to
  // the evaluated Longformer model rather than a paid Gradio Space.
  gradioSpaceUrl:
    'https://huggingface.co/anmol-unitmole/longformer-qasper-document-qa',

  staticSpaceUrl:
    'https://huggingface.co/spaces/anmol-unitmole/long-document-question-answering-longformer',

  modelCardUrl:
    'https://huggingface.co/anmol-unitmole/longformer-qasper-document-qa',
});