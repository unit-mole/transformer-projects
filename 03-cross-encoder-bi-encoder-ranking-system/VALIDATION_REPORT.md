# Validation Report

## Completed locally

- Browser project structure: passed
- Static Space metadata: passed
- Browser documents: 24
- Browser queries: 12
- Browser qrels: 36
- JavaScript syntax checks: passed
- Browser metric tests: 3 passed
- Python source compilation: passed
- Gradio application import: passed
- Python tests: 7 passed
- Python tests download Transformer models: no
- Model training during app startup: no

## Production Vite build

The build environment used to generate this downloadable bundle did not provide
package-registry access, so `npm install` and the final Vite build could not be
executed here.

The GitHub workflow performs:

```bash
npm install
npm run check
npm test
npm run build
python scripts/validate_dist.py
```

Run the same commands locally after extracting the files.

## Transformer execution

Actual Python MiniLM inference requires installing `requirements.txt` and
downloading the models.

Actual browser inference requires opening the Vite application with internet
access so Transformers.js can download the q8 ONNX model assets.

No model metric has been invented. Offline output files remain `status:
not_run` until the evaluation scripts are executed. Browser metrics are
calculated live only for labelled sample queries.
