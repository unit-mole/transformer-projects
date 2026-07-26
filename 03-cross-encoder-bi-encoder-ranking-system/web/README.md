# DocRank360 Static Frontend

This folder is the free Hugging Face Static Space deployment layer for Project
03.

## Technology

- Vite
- Vanilla JavaScript modules
- Transformers.js
- ONNX Runtime Web
- `Xenova/all-MiniLM-L6-v2`
- `Xenova/ms-marco-MiniLM-L-6-v2`

## Structure

```text
web/
├── index.html
├── package.json
├── vite.config.js
├── public/
│   ├── README.md
│   └── data/
└── src/
    ├── constants.js
    ├── data-loader.js
    ├── metrics.js
    ├── export-results.js
    ├── ranking-engine.js
    ├── ui.js
    ├── main.js
    └── styles.css
```

## Local development

```bash
npm install
npm run dev
```

Open the URL shown by Vite.

## Production build

```bash
npm run check
npm run build
npm run preview
```

The deployable Static Space is generated under:

```text
web/dist/
```

Vite copies `public/README.md` and `public/data/` to the build root so the
result can be uploaded directly to a Hugging Face Static Space.
