export const MODEL_IDS = Object.freeze({
  biEncoder: "Xenova/all-MiniLM-L6-v2",
  crossEncoder: "Xenova/ms-marco-MiniLM-L-6-v2",
});

export const DISPLAY_LIMITS = Object.freeze({
  minimumQueryCharacters: 3,
  maximumQueryCharacters: 500,
  maximumCandidateK: 20,
});

export const PROJECT_LINKS = Object.freeze({
  github: "https://github.com/unit-mole/transformer-projects",
  huggingFaceProfile: "https://huggingface.co/anmol-unitmole",
});

export const DEPLOYMENT = Object.freeze({
  platform: "Hugging Face Static Space",
  runtime: "Transformers.js 3.8.1 + ONNX Runtime Web",
  dtype: "q8",
});
