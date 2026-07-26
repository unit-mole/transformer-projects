# Portfolio Positioning

## One-line description

Built a two-stage Transformer search engine using MiniLM bi-encoder retrieval
and MS MARCO cross-encoder reranking, evaluated with Recall@K, MRR@10, nDCG@10,
rank movement, and latency, and deployed through Gradio on Hugging Face Spaces.

## GitHub pinned-repository description

Production-style information-retrieval portfolio project with semantic candidate
retrieval, cross-encoder reranking, graded qrels, ranking metrics, latency
benchmarking, responsible-use controls, tests, Docker, CI, and a live Gradio demo.

## Recruiter talking points

- explains why modern search and RAG use multi-stage retrieval;
- demonstrates both representation learning and pairwise relevance modeling;
- measures quality rather than using generic classification accuracy;
- separates cold start, retrieval, reranking, and end-to-end latency;
- includes failure analysis and responsible-use controls;
- connects naturally to quality complaint search, root-cause history, and
  corrective-action knowledge retrieval.

## Screenshots

- app overview;
- candidate retrieval table;
- reranked table with rank movement;
- latency JSON;
- metric comparison;
- latency-by-top-K;
- successful and failed query cases.
