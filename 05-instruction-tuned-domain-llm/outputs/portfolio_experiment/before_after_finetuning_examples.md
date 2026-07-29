# Before vs After LoRA Fine-Tuning

These examples are selected by held-out BERTScore improvement. Review them manually before making qualitative claims.

## recurrent neural network — `mlds-ext-e1bef5712e5d`

**Instruction:** Give an interview-style answer explaining recurrent neural network.

**Reference:** A strong interview answer is: A recurrent neural network updates a hidden state while processing a sequence. It models order and context in text or time-series data. For example, processing a sequence of sensor readings. I would also mention that standard rnns can struggle with long dependencies and vanishing gradients.

**Base FLAN-T5 response:**

ML/DS

**LoRA response:**

An interview answer is: Recurrent neural network is a recurrent neural network. It is a recurrent neural network. It is a recurrent neural network. It is a recurrent network.

**BERTScore F1 change:** 0.2006

---

## retrieval-augmented generation — `mlds-ext-2f78f3c3e743`

**Instruction:** Give an interview-style answer explaining retrieval-augmented generation.

**Reference:** A strong interview answer is: Retrieval-augmented generation retrieves relevant sources and includes them in the model context before generation. It can improve grounding and make answers traceable to documents. For example, answering questions about a portfolio by retrieving README passages. I would also mention that poor retrieval or ignored evidence can still produce incorrect answers.

**Base FLAN-T5 response:**

ML/DS

**LoRA response:**

An interview answer is: retrieval-augmented generation is a process that requires a lot of data. For example, a retrieval-augmented generation requires a lot of data. For example, a retrieval-augmented generation requires a lot of data. For example, retrieval-augmented generation requires a lot of data.

**BERTScore F1 change:** 0.1926

---

## class imbalance — `mlds-ext-5206f35df89f`

**Instruction:** Give an interview-style answer explaining class imbalance.

**Reference:** A strong interview answer is: Class imbalance means some labels occur much less often than others. A model can appear accurate while performing poorly on the minority class. For example, defects representing 2 percent of all inspected units. I would also mention that use class-aware metrics, suitable sampling or weighting, and threshold analysis.

**Base FLAN-T5 response:**

ML/DS

**LoRA response:**

A good interview answer is: Class imbalance is a result of class imbalance. Class imbalance is a result of class imbalance. Class imbalance is a result of class imbalance.

**BERTScore F1 change:** 0.1851

---

## model drift — `mlds-ext-261a67980f43`

**Instruction:** Give an interview-style answer explaining model drift.

**Reference:** A strong interview answer is: Model drift is a decline or change in model behavior after deployment. Relationships between inputs and outcomes can evolve. For example, a defect classifier becoming less accurate after a product redesign. I would also mention that drift detection requires timely labels or meaningful proxies.

**Base FLAN-T5 response:**

ML/DS

**LoRA response:**

An interview answer is: Model drift is a model that can be used in a model. It is a model that can be used in a model. It is a model that can be used in a model. It can be used in a model.

**BERTScore F1 change:** 0.1729

---

## self-supervised learning — `mlds-ext-ec9112efcbed`

**Instruction:** Give an interview-style answer explaining self-supervised learning.

**Reference:** A strong interview answer is: Self-supervised learning creates training signals from the data itself, such as predicting masked tokens. It enables representation learning before a smaller supervised fine-tuning stage. For example, pretraining a language model on unlabeled technical text and then adapting it to ticket classification. I would also mention that pretraining objectives do not guarantee downstream factual accuracy.

**Base FLAN-T5 response:**

ML/DS

**LoRA response:**

An interview answer is: Self-supervised learning is a learning tool that can be used in a variety of contexts. For example, self-supervised learning can be used in a variety of contexts. For example, self-supervised learning can be used in a variety of contexts. For example, self-supervised learning can be used in a variety of contexts.

**BERTScore F1 change:** 0.1650

---

## semantic search — `mlds-ext-b61bc960dfec`

**Instruction:** Give an interview-style answer explaining semantic search.

**Reference:** A strong interview answer is: Semantic search retrieves items by meaning rather than exact keyword overlap. It can match paraphrases and related concepts. For example, finding a project about defect detection from the query 'quality issue classifier'. I would also mention that embedding quality, chunking, indexing, and evaluation determine usefulness.

**Base FLAN-T5 response:**

ML/DS

**LoRA response:**

A good interview answer is: semantic search is a way to identify a particular object. For example, if a specific object is a model of a particular object. For example, if a specific object is a model of a specific object. For example, if a specific object is a model of a specific object. For example, if a specific object is a model of a specific object.

**BERTScore F1 change:** 0.1644

---

## quantization — `mlds-ext-0130527214a7`

**Instruction:** Give an interview-style answer explaining quantization.

**Reference:** A strong interview answer is: Quantization stores or computes model values with lower precision. It can reduce memory, download size, and inference latency. For example, using a four- or eight-bit ONNX model in a browser demo. I would also mention that lower precision can affect quality and hardware compatibility.

**Base FLAN-T5 response:**

ML/DS

**LoRA response:**

An interview answer is: Quantization is a measure of a model's value. It is a measure of a model's value. It is a measure of a model's value. It is a measure of a value.

**BERTScore F1 change:** 0.1561

---

## MAE vs RMSE — `mlds-ext-194e4c01718a`

**Instruction:** Compare MAE and RMSE.

**Reference:** MAE averages absolute errors, while RMSE squares errors before averaging and therefore emphasizes large misses. Use MAE for a robust, directly interpretable average error. Use RMSE when large errors should receive extra penalty. Both depend on target scale and should be paired with residual analysis.

**Base FLAN-T5 response:**

ML/DS

**LoRA response:**

MAE and RMSE are a two-dimensional model that can be used in a multi-dimensional model. They can be used in a multi-dimensional model. They can be used in a multi-dimensional model.

**BERTScore F1 change:** 0.1294

---

## error-analysis — `mlds-ext-746144a78be4`

**Instruction:** What should an error analysis include?

**Reference:** Segment errors by class, confidence, data source, time period, and important subgroups. Read representative false positives and false negatives, identify recurring causes, check annotation quality and leakage, and convert findings into data, feature, threshold, or modeling improvements.

**Base FLAN-T5 response:**

ML/DS

**LoRA response:**

Error analysis should be able to identify a specific error problem. It should be able to identify a specific error problem. It should be able to identify an error problem.

**BERTScore F1 change:** 0.1030

---

## macro-weighted-f1 — `mlds-ext-e9c769888e6b`

**Instruction:** Explain macro F1 and weighted F1.

**Reference:** Macro F1 averages class-level F1 scores equally, so rare classes matter as much as common ones. Weighted F1 weights each class by its support, so common classes influence the result more. Report per-class metrics as well when minority performance is important.

**Base FLAN-T5 response:**

Describe the ML/DS and ML/DS.

**LoRA response:**

macro F1 weighted F1 weighted F1 weighted F1 weighted F1 weighted F1 weighted F1 weighted F1 weighted F1 weighted F1.

**BERTScore F1 change:** 0.0825

---

## accuracy and F1-score — `mlds-ext-9a3f3c0f955e`

**Instruction:** Compare accuracy and F1-score.

**Reference:** Accuracy measures the overall proportion of correct predictions. F1 balances precision and recall for the positive class. Accuracy is reasonable when classes and error costs are balanced; F1 is usually more informative when the positive class is rare or both major error types matter.

**Base FLAN-T5 response:**

You are an educational ML and Data Science learning assistant

**LoRA response:**

Accurate accuracy and F1-score is a measure of the accuracy and reliability of a model. For example, accuracy is a measure of the accuracy and reliability of a model. For example, accuracy is a measure of accuracy and reliability of a model.

**BERTScore F1 change:** 0.0775

---

## model drift — `mlds-ext-1e0714d26ffc`

**Instruction:** Give a practical ML example of model drift.

**Reference:** Practical example: a defect classifier becoming less accurate after a product redesign. Why it matters: Relationships between inputs and outcomes can evolve. Key caution: Drift detection requires timely labels or meaningful proxies.

**Base FLAN-T5 response:**

ML/DS is a ML/DS learning assistant. ML/DS is a ML/DS learning assistant.

**LoRA response:**

Practical example: a model drift is a model that can be used as a model based model. It can be used as a model based model. It can be used as a model based model.

**BERTScore F1 change:** 0.0682

---
