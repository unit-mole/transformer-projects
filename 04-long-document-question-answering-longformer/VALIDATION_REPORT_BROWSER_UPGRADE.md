# Validation Report — Browser Deployment Upgrade

## Completed successfully

- Existing Python test suite: **10 tests passed**.
- Browser utility test suite: **8 tests passed**.
- JavaScript syntax validation passed for all source modules.
- Python source compilation and lightweight imports were already covered by the
  existing Project 04 validation workflow.
- The dedicated GitHub Actions workflow now validates both Python and browser
  components.

## Not executed in this artifact environment

The environment used to assemble this package could not complete external NPM
package downloads. Therefore, `npm run build` was not executed here. The source
uses pinned dependencies and the workflow/Static Space build will run:

```bash
npm install --no-audit --no-fund
npm test
npm run check
npm run build
```

Browser model inference was also not executed because it requires downloading
the ONNX model from Hugging Face. No browser accuracy or latency values are
invented or committed.

## Commands to run locally

```bash
cd 04-long-document-question-answering-longformer
pytest

cd web
npm install
npm test
npm run check
npm run build
npm run preview
```
