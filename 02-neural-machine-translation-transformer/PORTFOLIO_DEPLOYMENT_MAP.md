# Project 02 Portfolio Deployment Map

```text
GitHub repository
└── Complete Python ML engineering project
    ├── MarianMT inference pipelines
    ├── data preparation and language detection
    ├── SacreBLEU, chrF, latency and error analysis
    ├── tests, notebooks and GitHub Actions
    └── local Gradio interface for development

Hugging Face Model Hub
└── Model lineage and artifacts
    ├── Base models: Helsinki-NLP/opus-mt-en-hi and opus-mt-hi-en
    ├── Browser ONNX models: Xenova/opus-mt-en-hi and opus-mt-hi-en
    └── Publish personal model repos only after genuine fine-tuning or conversion

Hugging Face Static Space
└── web/
    ├── real browser-side MarianMT inference
    ├── automatic language detection
    ├── sentence and CSV batch translation
    ├── token preview, latency and confidence proxy
    └── no paid Gradio/Docker compute and no inference API
```

The Static Space is a deployment layer. It does not replace the Python project;
it makes the trained Transformer behavior directly testable while the GitHub
repository provides the deeper engineering and evaluation evidence.
