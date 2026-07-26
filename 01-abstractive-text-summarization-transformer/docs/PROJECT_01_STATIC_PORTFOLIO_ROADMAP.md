# Project 01 Static Portfolio Roadmap

## Static deployment does not reduce the project value

The deployment method changes the execution location:

- Python/Gradio: PyTorch inference runs in a Python process.
- Static/Transformers.js: quantized ONNX inference runs in the visitor's browser.

Both paths execute a real DistilBART encoder-decoder Transformer. The Static Space is not a mock user interface and does not depend on a paid inference endpoint.

## Recommended three-part structure

| Component | Portfolio purpose |
|---|---|
| GitHub repository | Full Python project, evaluation, tests, notebooks, baselines, and engineering structure |
| Hugging Face model references | Honest attribution of the Python checkpoint and browser ONNX conversion |
| Hugging Face Static Space | Live interactive summarization with browser inference |

## Keep the Python project

Retain:

```text
app.py
gradio_app.py
src/
scripts/
tests/
notebooks/
outputs/
requirements.txt
MODEL_CARD.md
```

These demonstrate PyTorch, Hugging Face Transformers, preprocessing, long-document handling, ROUGE, BERTScore, latency, baselines, LSTM comparison, automated tests, and error analysis.

## Add the browser deployment layer

```text
web/
├── README.md
├── index.html
├── package.json
├── vite.config.js
├── public/evaluation-results.json
├── src/
│   ├── main.js
│   ├── model-worker.js
│   ├── summarizer-client.js
│   ├── samples.js
│   ├── text-utils.js
│   └── styles.css
└── tests/text-utils.test.js
```

## Static demo evidence

The live demo should visibly present:

- Python and browser model names;
- encoder-decoder architecture;
- token counts and token-ID preview;
- model download progress;
- WebGPU/WASM runtime selection;
- minimum/maximum token controls;
- beam count, length penalty, and repetition controls;
- token-aware long-document chunking;
- generated summary;
- latency, compression ratio, word count, and chunk count;
- beam 1 versus selected-beam comparison;
- evaluation status without fabricated values;
- Transformer-versus-LSTM explanation;
- limitations and responsible use;
- links to GitHub and the public base models.

## Model repository strategy

Use the public checkpoints honestly:

```text
sshleifer/distilbart-cnn-12-6
Xenova/distilbart-cnn-12-6
```

Create a personal model repository only after actually fine-tuning or converting a model. When that occurs, publish a model card containing the base checkpoint, dataset, preprocessing, training details, measured metrics, examples, limitations, and intended use.

## Final project setup

```text
GitHub
└── Complete Python ML project + browser source

Hugging Face Model Hub
└── Honest base-model references or actual fine-tuned artifacts

Hugging Face Static Space
└── web/ deployed as a free interactive Transformers.js demo
```

This structure demonstrates Transformer inference, generative NLP, Python engineering, ONNX/browser deployment, evaluation, testing, frontend engineering, and CI/CD without requiring paid Hugging Face compute.
