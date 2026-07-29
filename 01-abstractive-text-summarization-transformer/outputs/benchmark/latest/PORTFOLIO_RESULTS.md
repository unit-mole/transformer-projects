# Project 01 Portfolio Benchmark Results

| model                 |   samples |   rouge1 |   rouge2 |   rougeL |   bertscore_f1 |   average_inference_seconds |   p95_inference_seconds |
|:----------------------|----------:|---------:|---------:|---------:|---------------:|----------------------------:|------------------------:|
| Lead-3                |       500 |   0.4008 |   0.1718 |   0.2468 |         0.2263 |                      0.0002 |                  0.0004 |
| TextRank              |       500 |   0.3489 |   0.1363 |   0.2289 |         0.185  |                      0.0014 |                  0.0024 |
| Pretrained DistilBART |       500 |   0.4469 |   0.2155 |   0.3082 |         0.3127 |                      0.3674 |                  0.4861 |
| Fine-tuned DistilBART |       500 |   0.4386 |   0.2062 |   0.3003 |         0.3416 |                      0.328  |                  0.4422 |
