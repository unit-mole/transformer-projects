# Automated Hallucination-Risk Triage

This report flags low reference support, unsupported numeric claims, unsupported attributions, and overconfident language. It is not a factuality verdict; complete the manual review template.

## Base model

Flagged examples: **27 / 36**

### mlds-ext-9a3f3c0f955e — accuracy and F1-score
- Risk types: `['low_reference_support']`
- Reference support: `0.0772`
- Generated response: You are an educational ML and Data Science learning assistant

### mlds-ext-194e4c01718a — MAE vs RMSE
- Risk types: `['low_reference_support']`
- Reference support: `0.1063`
- Generated response: ML/DS

### mlds-ext-04aabbbac1ba — LoRA configuration
- Risk types: `['low_reference_support']`
- Reference support: `0.1389`
- Generated response: You are an educational ML and Data Science learning assistant.

### mlds-ext-7c901d2b9a9b — embedding
- Risk types: `['low_reference_support']`
- Reference support: `0.2059`
- Generated response: Describe ML/DS as a learning assistant.

### mlds-ext-58d17c7da556 — quantization
- Risk types: `['low_reference_support']`
- Reference support: `0.0657`
- Generated response: Use a ML/DS learning assistant. Use ML/DS learning assistants.

### mlds-ext-cfc17cfc3f66 — retrieval-augmented generation
- Risk types: `['low_reference_support']`
- Reference support: `0.2747`
- Generated response: Describe ML/DS as a learning assistant.

### mlds-ext-636bfdc6ffb6 — self-supervised learning
- Risk types: `['low_reference_support']`
- Reference support: `0.0502`
- Generated response: Do not provide legal, medical, or safety-critical advice.

### mlds-ext-981c55ca07dc — semantic search
- Risk types: `['low_reference_support']`
- Reference support: `0.0453`
- Generated response: You are an educational ML/DS learning assistant. Answer clearly, stay within ML/DS topics, state uncertainty, and do not provide legal, medical, financial, immigration, or safety-critical advice.

### mlds-ext-d89e0e561886 — class imbalance
- Risk types: `['low_reference_support']`
- Reference support: `0.1399`
- Generated response: ML/DS is a ML/DS learning assistant. ML/DS is a ML/DS learning assistant.

### mlds-ext-1e0714d26ffc — model drift
- Risk types: `['low_reference_support']`
- Reference support: `0.1229`
- Generated response: ML/DS is a ML/DS learning assistant. ML/DS is a ML/DS learning assistant.

### mlds-ext-48e53e7aa70a — quantization
- Risk types: `['low_reference_support']`
- Reference support: `0.0451`
- Generated response: You are an ML/DS learning assistant. You are an ML/DS learning assistant. You are an ML/DS learning assistant. You are an ML/DS learning assistant.

### mlds-ext-f3523a1ac32e — retrieval-augmented generation
- Risk types: `['low_reference_support']`
- Reference support: `0.3992`
- Generated response: ML/DS is a method of retrieval-augmented generation.

### mlds-ext-96422ba70df9 — self-supervised learning
- Risk types: `['low_reference_support']`
- Reference support: `0.2078`
- Generated response: You are an ML/DS learning assistant. Answer clearly, stay within ML/DS topics, state uncertainty, and do not provide legal, medical, financial, immigration, or safety-critical advice.

### mlds-ext-e3cae1e5cccb — semantic search
- Risk types: `['low_reference_support']`
- Reference support: `0.1023`
- Generated response: You are an educational ML/DS learning assistant. Answer clearly, stay within ML/DS topics, state uncertainty, and do not provide legal, medical, financial, immigration, or safety-critical advice.

### mlds-ext-5206f35df89f — class imbalance
- Risk types: `['low_reference_support']`
- Reference support: `0.1360`
- Generated response: ML/DS

### mlds-ext-261a67980f43 — model drift
- Risk types: `['low_reference_support']`
- Reference support: `0.0517`
- Generated response: ML/DS

### mlds-ext-0130527214a7 — quantization
- Risk types: `['low_reference_support']`
- Reference support: `0.1768`
- Generated response: ML/DS

### mlds-ext-e1bef5712e5d — recurrent neural network
- Risk types: `['low_reference_support']`
- Reference support: `0.0996`
- Generated response: ML/DS

### mlds-ext-2f78f3c3e743 — retrieval-augmented generation
- Risk types: `['low_reference_support']`
- Reference support: `0.0886`
- Generated response: ML/DS

### mlds-ext-ec9112efcbed — self-supervised learning
- Risk types: `['low_reference_support']`
- Reference support: `0.0626`
- Generated response: ML/DS

## LoRA adapter

Flagged examples: **7 / 36**

### mlds-ext-d89e0e561886 — class imbalance
- Risk types: `['low_reference_support']`
- Reference support: `0.2466`
- Generated response: Practical example: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance: class imbalance

### mlds-ext-48e53e7aa70a — quantization
- Risk types: `['low_reference_support']`
- Reference support: `0.2193`
- Generated response: Practical example: quantization is a measure of the size of a sample. It is a measure of the size of a sample. It is a measure of the size of a sample.

### mlds-ext-f3523a1ac32e — retrieval-augmented generation
- Risk types: `['low_reference_support']`
- Reference support: `0.3649`
- Generated response: Practical example: retrieval-augmented generation is an example of retrieval-augmented generation. It is an example of retrieval-augmented generation. It is an example of retrieval-augmented generation. It is an example of retrieval-augmented generation.

### mlds-ext-96422ba70df9 — self-supervised learning
- Risk types: `['low_reference_support']`
- Reference support: `0.4090`
- Generated response: Practical example: self-supervised learning is an example of self-supervised learning. For example: self-supervised learning is an example of self-supervised learning. For example: self-supervised learning is an example of self-supervised learning. For example: self-supervised learning is an example of self-supervised learning.

### mlds-ext-57360469e319 — quality assistant grounding
- Risk types: `['low_reference_support']`
- Reference support: `0.2273`
- Generated response: A quality-learning assistant should be able to measure the performance of a dataset. For example, if a dataset is based on a dataset, it should be able to measure the performance of a dataset. For example, if a dataset is based on a dataset, it should be able to measure the performance of a dataset.

### mlds-ext-9606de835323 — reproducible model release
- Risk types: `['low_reference_support']`
- Reference support: `0.2743`
- Generated response: Using a reproducible LoRA model release, a reproducible LoRA model release should be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to be able to

### mlds-ext-c333a9d3e751 — human evaluation of generated answers
- Risk types: `['low_reference_support']`
- Reference support: `0.3728`
- Generated response: Practical workflow for human evaluation of an educational LLM. It is based on a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of a sample of
