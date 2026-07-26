# Data Documentation

## Bundled Samples

- `sample_articles.csv`: five original demonstration articles.
- `sample_summaries.csv`: the same articles with human-written reference summaries.
- `lstm_comparison_template.csv`: schema for actual LSTM Seq2Seq predictions.

These samples were created specifically for the portfolio and are safe to redistribute. They are not the CNN/DailyMail or XSum dataset and must not be described as benchmark data.

## Supported Public Datasets

`src/dataset_loader.py` supports:

- `EdinburghNLP/xsum` with `document` and `summary` columns.
- `abisee/cnn_dailymail`, configuration `3.0.0`, with `article` and `highlights` columns.
- Custom CSV files with common article/summary column names.

## Safe Repository Practice

Do not commit full datasets, private company reports, complaint narratives containing personal data, confidential quality records, or copyrighted articles without redistribution rights. Use a bounded public subset through the loader and record the dataset revision for reproducibility.
