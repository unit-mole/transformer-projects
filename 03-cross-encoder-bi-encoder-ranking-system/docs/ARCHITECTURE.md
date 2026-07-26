# System Architecture

## Three execution surfaces

### Python reference implementation

```text
CSV data
  ↓
shared preprocessing
  ↓
SentenceTransformer embeddings
  ↓
NumPy cosine index
  ↓
CrossEncoder reranking
  ↓
evaluation scripts and Gradio app
```

### Browser implementation

```text
Static JSON data
  ↓
Transformers.js feature-extraction pipeline
  ↓
browser document embedding cache
  ↓
cosine retrieval
  ↓
Transformers.js sequence-classification model
  ↓
live ranking tables and metrics
```

### Model Hub documentation

```text
base-model attribution
  + pipeline configuration
  + evaluation results
  + limitations
  + intended use
```

## Model mapping

| Purpose | Python | Browser |
|---|---|---|
| Bi-encoder | `sentence-transformers/all-MiniLM-L6-v2` | `Xenova/all-MiniLM-L6-v2` |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `Xenova/ms-marco-MiniLM-L-6-v2` |

## Why scores are not blended

Bi-encoder cosine similarities and cross-encoder logits are generated on
different scales. This project uses the bi-encoder strictly for candidate
generation and the cross-encoder strictly for reranking. A weighted score blend
would require calibration and validation.

## Browser security and privacy

- no user text is sent to a project-owned server;
- model files are downloaded from Hugging Face;
- inference runs in browser memory;
- the demo does not persist user queries;
- exported JSON is generated locally;
- users must not enter confidential or personal text.
