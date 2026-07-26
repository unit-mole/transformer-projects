# Hugging Face Deployment Card — Project 01

Project 01 uses a dedicated browser application under `web/` for the free Hugging Face Static Space.

## Deployment source

```text
web/
├── README.md
├── index.html
├── package.json
├── vite.config.js
├── public/
├── src/
└── tests/
```

The deployment workflow uploads the contents of `web/` to the root of the Space repository. The actual Space card and required YAML metadata are in `web/README.md`.

## Runtime

- Space SDK: `static`
- Build command: `npm run build`
- Served file: `dist/index.html`
- Browser library: `@huggingface/transformers==3.8.1`
- Browser model: `Xenova/distilbart-cnn-12-6`
- Preferred runtime: WebGPU with `q4f16`
- Fallback runtime: WASM with `q8`
- Python server: none
- Paid Hugging Face compute: none

## Honest model attribution

The browser model is an ONNX conversion of `sshleifer/distilbart-cnn-12-6`. It is used as a public base model and is not presented as a model trained by the portfolio author.

## Responsible use

Summaries can omit, distort, or hallucinate details. Do not paste sensitive or confidential text. Human review is mandatory before real-world use.

## Links

- GitHub: https://github.com/unit-mole/transformer-projects
- Project: https://github.com/unit-mole/transformer-projects/tree/main/01-abstractive-text-summarization-transformer
- Browser model: https://huggingface.co/Xenova/distilbart-cnn-12-6
