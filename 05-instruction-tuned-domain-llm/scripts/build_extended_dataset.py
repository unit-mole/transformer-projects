#!/usr/bin/env python
"""Build the expanded, public-safe ML/Data Science instruction dataset.

The generator combines the original hand-authored curriculum with a larger
curated knowledge bank. Splits are assigned by topic group so that paraphrases
of the same topic cannot leak across train, validation, and test sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_DIR / "data" / "ml_ds_instruction_dataset.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "ml_ds_instruction_dataset_extended.jsonl"
DEFAULT_EVAL = PROJECT_DIR / "data" / "evaluation_prompts_extended.jsonl"
DEFAULT_STATS = PROJECT_DIR / "outputs" / "extended_dataset_statistics.json"
DEFAULT_VALIDATION = PROJECT_DIR / "outputs" / "extended_dataset_validation_report.json"

SOURCE_LABEL = "self-authored and curated public-safe ML/DS curriculum"

CONCEPTS: list[dict[str, str]] = [
    {"topic": "supervised learning", "definition": "Supervised learning learns a mapping from labeled inputs to known targets.", "why": "It is used when historical examples contain the outcome you want to predict.", "example": "predicting whether a quality case will require escalation from past labeled cases.", "caveat": "Labels must represent the real decision and must be available without leakage."},
    {"topic": "unsupervised learning", "definition": "Unsupervised learning looks for structure in data without a target label.", "why": "It can reveal clusters, compressed representations, or unusual observations.", "example": "grouping similar defect descriptions before a labeling project.", "caveat": "The discovered groups still need domain interpretation."},
    {"topic": "semi-supervised learning", "definition": "Semi-supervised learning combines a small labeled set with a larger unlabeled set.", "why": "It is useful when labels are expensive but raw data is plentiful.", "example": "using a few reviewed quality tickets and many unreviewed tickets to improve classification.", "caveat": "Incorrect pseudo-labels can reinforce model errors."},
    {"topic": "self-supervised learning", "definition": "Self-supervised learning creates training signals from the data itself, such as predicting masked tokens.", "why": "It enables representation learning before a smaller supervised fine-tuning stage.", "example": "pretraining a language model on unlabeled technical text and then adapting it to ticket classification.", "caveat": "Pretraining objectives do not guarantee downstream factual accuracy."},
    {"topic": "classification", "definition": "Classification predicts one category from a fixed set of labels.", "why": "It is suitable for outcomes such as pass/fail, defect type, or escalation class.", "example": "classifying a complaint as mechanical, chemical, software, or packaging related.", "caveat": "Class probabilities and threshold choices matter when error costs differ."},
    {"topic": "regression", "definition": "Regression predicts a continuous numeric value.", "why": "It is useful for quantities such as demand, cycle time, or expected case volume.", "example": "forecasting the number of incoming quality cases next month.", "caveat": "Inspect residuals and use metrics that match the business cost of errors."},
    {"topic": "clustering", "definition": "Clustering groups observations so that items in the same group are more similar than items in other groups.", "why": "It can organize unlabeled data and reveal recurring patterns.", "example": "discovering recurring themes in free-text complaint summaries.", "caveat": "Cluster labels are not ground truth and depend on features and distance choices."},
    {"topic": "dimensionality reduction", "definition": "Dimensionality reduction represents data with fewer variables while preserving useful structure.", "why": "It can aid visualization, compression, noise reduction, and downstream modeling.", "example": "projecting hundreds of text-embedding dimensions into two dimensions for exploration.", "caveat": "A compact representation can discard information important for prediction."},
    {"topic": "overfitting", "definition": "Overfitting occurs when a model learns training-specific noise instead of general patterns.", "why": "It creates strong training performance but weak performance on unseen data.", "example": "a deep tree memorizing rare ticket identifiers.", "caveat": "Use proper validation, regularization, simpler models, and more representative data."},
    {"topic": "underfitting", "definition": "Underfitting occurs when a model is too simple or insufficiently trained to capture important patterns.", "why": "Both training and validation performance remain poor.", "example": "using a straight line for a strongly nonlinear relationship.", "caveat": "Better features, a richer model, or longer training may help."},
    {"topic": "bias-variance trade-off", "definition": "The bias-variance trade-off balances errors from an overly simple model against errors from sensitivity to the training sample.", "why": "Generalization often improves at an intermediate level of model complexity.", "example": "comparing a shallow tree, a tuned tree, and an unrestricted tree.", "caveat": "The best balance depends on data size, noise, and the evaluation target."},
    {"topic": "regularization", "definition": "Regularization discourages unnecessary model complexity.", "why": "It can reduce overfitting by constraining weights or model behavior.", "example": "using L2 weight decay in logistic regression or dropout in a neural network.", "caveat": "Too much regularization can cause underfitting."},
    {"topic": "cross-validation", "definition": "Cross-validation repeatedly trains and validates a model on different folds of the training data.", "why": "It provides a more stable estimate than a single split and supports model selection.", "example": "using stratified five-fold validation for an imbalanced classifier.", "caveat": "Time-series and grouped data require specialized split strategies."},
    {"topic": "data leakage", "definition": "Data leakage occurs when training features contain target information or information unavailable at prediction time.", "why": "It produces unrealistically high validation scores that fail in production.", "example": "using a field completed after a quality investigation to predict the investigation result.", "caveat": "Audit feature creation, timing, grouping, and preprocessing boundaries."},
    {"topic": "feature engineering", "definition": "Feature engineering transforms raw data into representations that make useful patterns easier to learn.", "why": "Good features can improve accuracy, stability, and interpretability.", "example": "creating rolling case counts, ratios, age buckets, and text-derived indicators.", "caveat": "Features must be reproducible and available at inference time."},
    {"topic": "standardization", "definition": "Standardization centers a numeric feature and scales it by its standard deviation.", "why": "It places features on comparable scales for models sensitive to magnitude.", "example": "scaling inputs before logistic regression or a support vector machine.", "caveat": "Fit the scaler on training data only."},
    {"topic": "normalization", "definition": "Normalization rescales values to a chosen range or scales each sample to a norm.", "why": "It can make distances or optimization more stable in selected applications.", "example": "L2-normalizing text embeddings before cosine similarity.", "caveat": "The word normalization is used differently across domains, so document the exact transformation."},
    {"topic": "one-hot encoding", "definition": "One-hot encoding creates one binary column for each category.", "why": "It lets many models use nominal categories without imposing an artificial order.", "example": "encoding product family or region before linear modeling.", "caveat": "High-cardinality columns can create many sparse features."},
    {"topic": "target encoding", "definition": "Target encoding replaces a category with a statistic computed from the target.", "why": "It can represent high-cardinality categories compactly.", "example": "encoding supplier identity using smoothed historical defect rate.", "caveat": "It is highly leakage-prone and must be computed within training folds."},
    {"topic": "missing-value handling", "definition": "Missing-value handling identifies why values are absent and applies a suitable strategy such as imputation, indicators, or model-native treatment.", "why": "Unmanaged missingness can break pipelines or introduce bias.", "example": "adding a missing indicator and median imputation for a sensor feature.", "caveat": "Missingness itself may carry information and can change over time."},
    {"topic": "class imbalance", "definition": "Class imbalance means some labels occur much less often than others.", "why": "A model can appear accurate while performing poorly on the minority class.", "example": "defects representing 2 percent of all inspected units.", "caveat": "Use class-aware metrics, suitable sampling or weighting, and threshold analysis."},
    {"topic": "probability calibration", "definition": "Probability calibration checks whether predicted probabilities match observed frequencies.", "why": "A calibrated 0.8 prediction should be correct about 80 percent of the time in comparable cases.", "example": "using calibrated escalation risk to prioritize review capacity.", "caveat": "Calibration can drift and should be checked on representative data."},
    {"topic": "threshold tuning", "definition": "Threshold tuning chooses the probability cutoff used to convert scores into decisions.", "why": "The default 0.5 threshold rarely reflects real false-positive and false-negative costs.", "example": "lowering the threshold when missing a critical defect is more costly than reviewing a false alert.", "caveat": "Tune on validation data and report final performance on untouched test data."},
    {"topic": "hyperparameter tuning", "definition": "Hyperparameter tuning searches settings chosen before training, such as depth, learning rate, or regularization strength.", "why": "Good settings can improve generalization and efficiency.", "example": "searching tree depth and minimum leaf size with cross-validation.", "caveat": "Repeated tuning can overfit the validation process."},
    {"topic": "random search", "definition": "Random search samples hyperparameter combinations from defined distributions.", "why": "It often explores useful regions more efficiently than a rigid grid.", "example": "sampling learning rates on a log scale and tree depths from a range.", "caveat": "Results depend on sensible search spaces and an adequate trial budget."},
    {"topic": "Bayesian optimization", "definition": "Bayesian optimization uses prior trials to choose promising hyperparameter settings.", "why": "It can reduce the number of expensive experiments needed.", "example": "tuning a gradient boosting model with an Optuna study.", "caveat": "Complex search does not replace a sound validation design."},
    {"topic": "ensemble learning", "definition": "Ensemble learning combines predictions from multiple models.", "why": "Diverse models can reduce variance, bias, or both.", "example": "averaging several regression models or stacking classifiers.", "caveat": "Ensembles add operational complexity and can reduce interpretability."},
    {"topic": "bagging", "definition": "Bagging trains models on resampled datasets and aggregates their predictions.", "why": "It mainly reduces variance for unstable learners.", "example": "random forest combines many bootstrapped decision trees.", "caveat": "Bagging does not automatically correct systematic bias."},
    {"topic": "boosting", "definition": "Boosting builds learners sequentially so later learners focus on previous errors.", "why": "It can create a strong model from many weak learners.", "example": "gradient boosting trees modeling residual errors stage by stage.", "caveat": "Aggressive boosting can overfit noisy data without regularization."},
    {"topic": "random forest", "definition": "Random forest averages many decision trees trained on bootstrap samples and random feature subsets.", "why": "It captures nonlinear interactions and is usually robust with limited tuning.", "example": "predicting case escalation from structured quality attributes.", "caveat": "Large forests can be less interpretable and probabilities may need calibration."},
    {"topic": "gradient boosting", "definition": "Gradient boosting adds trees sequentially to reduce a differentiable loss.", "why": "It is a strong choice for many structured-data problems.", "example": "predicting defect quantity with XGBoost or LightGBM.", "caveat": "Careful validation and tuning are needed to control overfitting."},
    {"topic": "linear regression", "definition": "Linear regression models a numeric target as a weighted sum of features.", "why": "It provides a transparent baseline and interpretable coefficients under suitable assumptions.", "example": "estimating cycle time from workload and process variables.", "caveat": "Check nonlinear patterns, residual behavior, multicollinearity, and outliers."},
    {"topic": "logistic regression", "definition": "Logistic regression models the log-odds of a class using a linear combination of features.", "why": "It is fast, interpretable, and often a strong classification baseline.", "example": "estimating the probability that a case will be overdue.", "caveat": "Nonlinear relationships need engineered features or a more flexible model."},
    {"topic": "decision tree", "definition": "A decision tree makes predictions through a sequence of feature-based splits.", "why": "It captures nonlinear interactions and is easy to visualize at small depth.", "example": "splitting cases by severity, age, and product family.", "caveat": "Unrestricted trees are unstable and prone to overfitting."},
    {"topic": "support vector machine", "definition": "A support vector machine finds a decision boundary with a large margin between classes.", "why": "Kernel functions can model nonlinear boundaries in moderate-sized datasets.", "example": "classifying text embeddings or standardized sensor features.", "caveat": "Training and probability estimation can be costly on large datasets."},
    {"topic": "k-nearest neighbors", "definition": "k-nearest neighbors predicts from the labels or values of nearby training examples.", "why": "It is simple and can model local patterns without fitting explicit parameters.", "example": "classifying a new measurement using nearby standardized observations.", "caveat": "It is sensitive to scale, irrelevant features, and inference cost."},
    {"topic": "naive Bayes", "definition": "Naive Bayes applies Bayes' rule with a conditional-independence assumption between features.", "why": "It is fast and often effective for sparse text features.", "example": "classifying complaint text using word counts.", "caveat": "The independence assumption is simplified and probabilities may be poorly calibrated."},
    {"topic": "principal component analysis", "definition": "Principal component analysis creates orthogonal directions that capture decreasing amounts of variance.", "why": "It can compress correlated numeric features and support visualization.", "example": "reducing dozens of correlated sensor readings before exploration.", "caveat": "High variance does not always mean high predictive value, and components can be hard to interpret."},
    {"topic": "k-means clustering", "definition": "K-means partitions observations around k centroids by minimizing within-cluster squared distance.", "why": "It is fast and useful when clusters are roughly compact and spherical.", "example": "segmenting products by standardized usage patterns.", "caveat": "It requires choosing k and is sensitive to scaling, initialization, and outliers."},
    {"topic": "DBSCAN", "definition": "DBSCAN forms clusters from dense regions and labels sparse points as noise.", "why": "It can find irregularly shaped clusters without choosing the number of clusters in advance.", "example": "detecting spatial groups and isolated anomalies.", "caveat": "The distance and density parameters are sensitive to scale and varying density."},
    {"topic": "anomaly detection", "definition": "Anomaly detection identifies observations that differ substantially from expected patterns.", "why": "It supports screening when rare abnormal cases have limited labels.", "example": "flagging unusual combinations of sensor readings for review.", "caveat": "An unusual point is not automatically an error or defect."},
    {"topic": "neural network", "definition": "A neural network composes layers of weighted transformations and nonlinear activations.", "why": "It can learn complex functions from large and varied data.", "example": "classifying images, sequences, or tabular signals.", "caveat": "It needs careful optimization, regularization, and evaluation."},
    {"topic": "activation function", "definition": "An activation function adds nonlinearity to a neural network layer.", "why": "Without nonlinear activations, stacked linear layers would remain a linear model.", "example": "using ReLU in hidden layers or sigmoid for a binary output.", "caveat": "The choice affects gradient flow and output interpretation."},
    {"topic": "backpropagation", "definition": "Backpropagation applies the chain rule to compute how each parameter affects the loss.", "why": "Optimizers use these gradients to update neural network weights.", "example": "propagating classification error from the output layer to earlier layers.", "caveat": "Gradient quality depends on the computation graph, scale, and numerical stability."},
    {"topic": "gradient descent", "definition": "Gradient descent updates parameters in the direction that reduces the loss.", "why": "Mini-batch variants make neural network optimization practical on large datasets.", "example": "Adam updating model weights after each batch.", "caveat": "Learning rate and optimizer settings strongly affect convergence."},
    {"topic": "batch normalization", "definition": "Batch normalization normalizes intermediate activations using batch statistics and learns a scale and shift.", "why": "It can stabilize and accelerate training in many neural networks.", "example": "placing batch normalization after a convolution.", "caveat": "Training and inference use different statistics, and very small batches can be problematic."},
    {"topic": "dropout", "definition": "Dropout randomly masks activations during training.", "why": "It discourages units from relying too heavily on one another and can reduce overfitting.", "example": "using dropout before a classification head.", "caveat": "Dropout is disabled during evaluation and excessive dropout can underfit."},
    {"topic": "convolutional neural network", "definition": "A convolutional neural network learns local filters that share weights across spatial positions.", "why": "This structure is efficient for images and other grid-like data.", "example": "detecting edges, textures, and objects in inspection images.", "caveat": "CNN performance still depends on data quality, resolution, and distribution shift."},
    {"topic": "recurrent neural network", "definition": "A recurrent neural network updates a hidden state while processing a sequence.", "why": "It models order and context in text or time-series data.", "example": "processing a sequence of sensor readings.", "caveat": "Standard RNNs can struggle with long dependencies and vanishing gradients."},
    {"topic": "long short-term memory network", "definition": "An LSTM is a recurrent network with gates that control stored, forgotten, and exposed information.", "why": "The gates help it learn longer dependencies than a basic RNN.", "example": "modeling a long sequence of process measurements.", "caveat": "LSTMs are sequential and can be slower to train than parallel Transformer models."},
    {"topic": "attention mechanism", "definition": "Attention computes weighted relationships between elements so a model can focus on relevant context.", "why": "It lets information flow directly between distant positions.", "example": "a translation model emphasizing source words relevant to the next target word.", "caveat": "Attention weights are useful diagnostics but are not always faithful explanations."},
    {"topic": "Transformer", "definition": "A Transformer uses attention, feed-forward layers, residual connections, and normalization to process sequences.", "why": "It supports parallel training and models long-range relationships effectively.", "example": "FLAN-T5 following a natural-language instruction.", "caveat": "Context length, compute cost, data quality, and hallucination remain important limitations."},
    {"topic": "positional encoding", "definition": "Positional encoding gives a Transformer information about token order.", "why": "Self-attention alone does not inherently know sequence position.", "example": "adding learned or relative position information to token representations.", "caveat": "Different architectures use different position mechanisms and extrapolate differently."},
    {"topic": "encoder-decoder Transformer", "definition": "An encoder-decoder Transformer encodes an input sequence and generates an output sequence conditioned on that representation.", "why": "It is well suited to translation, summarization, and text-to-text instruction tasks.", "example": "T5 converting an ML question into an educational answer.", "caveat": "Generation quality depends on decoding settings and training alignment."},
    {"topic": "tokenization", "definition": "Tokenization converts text into model vocabulary units and numeric IDs.", "why": "The model processes tokens rather than raw characters or words.", "example": "SentencePiece splitting an uncommon technical word into subword pieces.", "caveat": "Token counts affect context length, latency, and cost."},
    {"topic": "embeddings", "definition": "Embeddings are dense vectors that represent items such as words, sentences, users, or products.", "why": "Similar meaning or behavior can be reflected by geometric proximity.", "example": "retrieving project documentation with sentence embeddings and cosine similarity.", "caveat": "Similarity reflects the training objective and may encode bias."},
    {"topic": "transfer learning", "definition": "Transfer learning reuses knowledge from a pretrained model for a new task.", "why": "It reduces the data and compute needed compared with training from scratch.", "example": "fine-tuning a pretrained vision or language model on a smaller domain dataset.", "caveat": "The source and target domains must be compatible enough for useful transfer."},
    {"topic": "fine-tuning", "definition": "Fine-tuning updates a pretrained model using task- or domain-specific data.", "why": "It specializes general representations for a narrower objective.", "example": "adapting FLAN-T5 to answer ML and Data Science learning questions.", "caveat": "Small or biased datasets can cause overfitting or catastrophic forgetting."},
    {"topic": "instruction tuning", "definition": "Instruction tuning trains a model on instruction-response pairs so it follows natural-language tasks more reliably.", "why": "It aligns generation with user requests such as explaining, comparing, or producing examples.", "example": "teaching FLAN-T5 to answer multiple types of ML learning prompts.", "caveat": "Instruction following does not guarantee factual correctness."},
    {"topic": "LoRA", "definition": "LoRA adds trainable low-rank matrices to selected layers while leaving base weights frozen.", "why": "It reduces trainable parameters, optimizer memory, and artifact size.", "example": "adapting the query and value projections in FLAN-T5 attention.", "caveat": "Rank, target modules, data quality, and learning rate still require validation."},
    {"topic": "parameter-efficient fine-tuning", "definition": "Parameter-efficient fine-tuning adapts a model by updating a small subset of parameters or added modules.", "why": "It makes experimentation and model sharing more practical on limited hardware.", "example": "training a LoRA adapter instead of all FLAN-T5 weights.", "caveat": "Efficiency does not remove the need for rigorous evaluation."},
    {"topic": "quantization", "definition": "Quantization stores or computes model values with lower precision.", "why": "It can reduce memory, download size, and inference latency.", "example": "using a four- or eight-bit ONNX model in a browser demo.", "caveat": "Lower precision can affect quality and hardware compatibility."},
    {"topic": "retrieval-augmented generation", "definition": "Retrieval-augmented generation retrieves relevant sources and includes them in the model context before generation.", "why": "It can improve grounding and make answers traceable to documents.", "example": "answering questions about a portfolio by retrieving README passages.", "caveat": "Poor retrieval or ignored evidence can still produce incorrect answers."},
    {"topic": "semantic search", "definition": "Semantic search retrieves items by meaning rather than exact keyword overlap.", "why": "It can match paraphrases and related concepts.", "example": "finding a project about defect detection from the query 'quality issue classifier'.", "caveat": "Embedding quality, chunking, indexing, and evaluation determine usefulness."},
    {"topic": "vector database", "definition": "A vector database stores embeddings and supports nearest-neighbor search with metadata filtering.", "why": "It enables scalable semantic retrieval for RAG and recommendation systems.", "example": "storing portfolio document chunks and their source links.", "caveat": "A vector database is infrastructure, not a guarantee of relevant retrieval."},
    {"topic": "model drift", "definition": "Model drift is a decline or change in model behavior after deployment.", "why": "Relationships between inputs and outcomes can evolve.", "example": "a defect classifier becoming less accurate after a product redesign.", "caveat": "Drift detection requires timely labels or meaningful proxies."},
    {"topic": "data drift", "definition": "Data drift is a change in the distribution of model inputs.", "why": "It can signal new populations, processes, or data pipelines.", "example": "a major shift in product-family proportions after a launch.", "caveat": "Input drift does not always imply performance degradation."},
    {"topic": "MLOps", "definition": "MLOps applies engineering practices to build, release, monitor, and govern machine-learning systems.", "why": "It connects experiments to reproducible and maintainable production workflows.", "example": "versioning data, code, models, tests, deployment, and monitoring dashboards.", "caveat": "Tools should support a clear process rather than replace one."},
    {"topic": "model monitoring", "definition": "Model monitoring tracks input quality, predictions, latency, failures, drift, and outcomes after deployment.", "why": "Offline metrics do not guarantee continued real-world performance.", "example": "monitoring defect recall, prediction confidence, and feature missingness over time.", "caveat": "Alerts need ownership, thresholds, and response procedures."},
    {"topic": "model explainability", "definition": "Model explainability methods describe factors associated with a model prediction or behavior.", "why": "They support debugging, communication, and risk review.", "example": "using SHAP values to inspect features influencing escalation predictions.", "caveat": "An explanation of the model is not proof of causality."},
    {"topic": "SHAP", "definition": "SHAP assigns feature contributions based on cooperative game-theory ideas.", "why": "It provides local and aggregate views of model behavior.", "example": "showing which variables pushed a case risk score up or down.", "caveat": "Results depend on model, background data, feature dependence, and implementation choices."},
    {"topic": "reproducibility", "definition": "Reproducibility means another run can recreate the data processing, training settings, artifacts, and reported results.", "why": "It makes experiments auditable and comparable.", "example": "saving seeds, package versions, configuration, split IDs, and model checkpoints.", "caveat": "GPU operations can still contain nondeterminism unless explicitly controlled."},
    {"topic": "train-validation-test split", "definition": "A train-validation-test split separates model fitting, model selection, and final unbiased evaluation.", "why": "Keeping the test set untouched reduces optimistic reporting.", "example": "training on 80 percent, tuning on 10 percent, and reporting once on 10 percent.", "caveat": "Use grouped or time-aware splits when observations are related."},
    {"topic": "time-series validation", "definition": "Time-series validation trains on earlier periods and evaluates on later periods.", "why": "It matches the direction of future prediction and avoids future-to-past leakage.", "example": "training on January through September and validating on October.", "caveat": "Seasonality and multiple forecast horizons may require rolling windows."},
    {"topic": "experiment tracking", "definition": "Experiment tracking records configurations, code versions, metrics, artifacts, and notes for each run.", "why": "It enables fair comparison and prevents lost results.", "example": "saving LoRA rank, learning rate, dataset fingerprint, and test metrics under a run ID.", "caveat": "Tracked metadata must be complete and consistently named."},
    {"topic": "early stopping", "definition": "Early stopping ends training when validation performance stops improving.", "why": "It can reduce overfitting and unnecessary compute.", "example": "stopping after two validation checks without lower loss.", "caveat": "Patience and monitored metric should match the real objective."},
    {"topic": "beam search", "definition": "Beam search keeps several high-scoring partial sequences during generation.", "why": "It can improve deterministic sequence generation compared with greedy decoding.", "example": "using four beams for evaluation responses from FLAN-T5.", "caveat": "More beams increase latency and may favor generic output."},
    {"topic": "temperature sampling", "definition": "Temperature changes how sharply generation probabilities are distributed.", "why": "Lower values make output more deterministic, while higher values increase variation.", "example": "using temperature zero for repeatable evaluation and a small positive value for demos.", "caveat": "Sampling makes metric comparisons less reproducible unless seeds and settings are fixed."},
]

COMPARISONS: list[dict[str, str]] = [
    {"topic": "classification vs regression", "a": "classification", "b": "regression", "summary": "Classification predicts discrete labels, while regression predicts continuous numeric values.", "choose_a": "Use classification for outcomes such as defect type or escalation class.", "choose_b": "Use regression for quantities such as case volume or cycle time.", "caveat": "Ordinal or count targets may require specialized formulations."},
    {"topic": "precision vs recall", "a": "precision", "b": "recall", "summary": "Precision measures how many predicted positives are correct, while recall measures how many actual positives were found.", "choose_a": "Prioritize precision when false alarms are expensive.", "choose_b": "Prioritize recall when missed positives are expensive.", "caveat": "Choose thresholds using business costs and review the precision-recall curve."},
    {"topic": "MAE vs RMSE", "a": "MAE", "b": "RMSE", "summary": "MAE averages absolute errors, while RMSE squares errors before averaging and therefore emphasizes large misses.", "choose_a": "Use MAE for a robust, directly interpretable average error.", "choose_b": "Use RMSE when large errors should receive extra penalty.", "caveat": "Both depend on target scale and should be paired with residual analysis."},
    {"topic": "L1 vs L2 regularization", "a": "L1", "b": "L2", "summary": "L1 encourages sparse coefficients and can set some to zero, while L2 shrinks coefficients smoothly.", "choose_a": "Use L1 when sparse feature selection is valuable.", "choose_b": "Use L2 for stable shrinkage with correlated predictors.", "caveat": "Elastic Net combines both behaviors."},
    {"topic": "decision tree vs random forest", "a": "decision tree", "b": "random forest", "summary": "A single tree is easy to inspect but unstable, while a random forest averages many randomized trees for better robustness.", "choose_a": "Use a shallow tree when direct interpretability is central.", "choose_b": "Use a random forest when predictive stability matters more.", "caveat": "Both may need calibration and careful handling of high-cardinality variables."},
    {"topic": "random forest vs gradient boosting", "a": "random forest", "b": "gradient boosting", "summary": "Random forest builds trees largely independently and averages them, while gradient boosting builds trees sequentially to correct errors.", "choose_a": "Use random forest for a robust baseline with limited tuning.", "choose_b": "Use gradient boosting when maximum structured-data accuracy justifies more tuning.", "caveat": "Validate both because the best choice depends on data and constraints."},
    {"topic": "logistic regression vs decision tree", "a": "logistic regression", "b": "decision tree", "summary": "Logistic regression learns a linear decision boundary, while a tree uses nonlinear feature splits.", "choose_a": "Use logistic regression for a fast interpretable baseline and calibrated relationships.", "choose_b": "Use a tree for nonlinear interactions and rule-like explanations.", "caveat": "Regularization and depth control are important for fair comparison."},
    {"topic": "k-means vs DBSCAN", "a": "k-means", "b": "DBSCAN", "summary": "K-means assigns every point to one of k centroid-based clusters, while DBSCAN finds dense regions and can label points as noise.", "choose_a": "Use k-means for compact roughly spherical clusters when k is meaningful.", "choose_b": "Use DBSCAN for irregular shapes and explicit noise handling.", "caveat": "Both depend strongly on feature scaling and distance definitions."},
    {"topic": "PCA vs feature selection", "a": "PCA", "b": "feature selection", "summary": "PCA creates new combinations of features, while feature selection keeps a subset of original variables.", "choose_a": "Use PCA for compression of correlated numeric data.", "choose_b": "Use feature selection when preserving original feature meaning is important.", "caveat": "Neither method guarantees better predictive performance."},
    {"topic": "bagging vs boosting", "a": "bagging", "b": "boosting", "summary": "Bagging trains learners in parallel on resampled data, while boosting trains learners sequentially to address prior errors.", "choose_a": "Use bagging mainly to reduce variance.", "choose_b": "Use boosting to build a stronger learner through staged correction.", "caveat": "Boosting is often more sensitive to noise and tuning."},
    {"topic": "RNN vs LSTM", "a": "RNN", "b": "LSTM", "summary": "A basic RNN carries a hidden state, while an LSTM adds gates that improve long-range memory and gradient flow.", "choose_a": "Use a simple RNN for short sequences or educational baselines.", "choose_b": "Use an LSTM when longer dependencies matter.", "caveat": "Transformers may train faster through parallel processing on many sequence tasks."},
    {"topic": "LSTM vs Transformer", "a": "LSTM", "b": "Transformer", "summary": "LSTMs process sequences recurrently, while Transformers use attention to connect positions and support parallel training.", "choose_a": "Use LSTM for compact sequential baselines or limited data.", "choose_b": "Use Transformers for long-range context and pretrained transfer.", "caveat": "Compute, latency, sequence length, and deployment constraints matter."},
    {"topic": "CNN vs Vision Transformer", "a": "CNN", "b": "Vision Transformer", "summary": "CNNs encode local spatial inductive bias through convolutions, while Vision Transformers process image patches with attention.", "choose_a": "Use CNNs for efficient strong baselines and smaller datasets.", "choose_b": "Use Vision Transformers when pretrained models and larger-scale transfer are available.", "caveat": "Benchmark accuracy, latency, memory, and robustness on the target data."},
    {"topic": "encoder-only vs encoder-decoder Transformer", "a": "encoder-only Transformer", "b": "encoder-decoder Transformer", "summary": "Encoder-only models create contextual representations, while encoder-decoder models map an input sequence to a generated output sequence.", "choose_a": "Use encoder-only models for classification, embeddings, and extractive tasks.", "choose_b": "Use encoder-decoder models for translation, summarization, and text-to-text instruction following.", "caveat": "Decoder-only models provide another generative architecture with different deployment trade-offs."},
    {"topic": "full fine-tuning vs LoRA", "a": "full fine-tuning", "b": "LoRA", "summary": "Full fine-tuning updates all model weights, while LoRA trains small low-rank adapters in selected modules.", "choose_a": "Use full fine-tuning when compute, data, and storage allow and broad adaptation is needed.", "choose_b": "Use LoRA for efficient domain adaptation and small shareable artifacts.", "caveat": "Both require identical held-out evaluation for a fair comparison."},
    {"topic": "fine-tuning vs retrieval-augmented generation", "a": "fine-tuning", "b": "retrieval-augmented generation", "summary": "Fine-tuning changes model behavior through training, while RAG supplies retrieved knowledge at inference time.", "choose_a": "Use fine-tuning for style, task format, and stable domain behavior.", "choose_b": "Use RAG for changing facts, source citations, and document-specific answers.", "caveat": "Many strong systems combine both."},
    {"topic": "keyword search vs semantic search", "a": "keyword search", "b": "semantic search", "summary": "Keyword search matches literal terms, while semantic search matches vector representations of meaning.", "choose_a": "Use keyword search for exact identifiers and transparent term matching.", "choose_b": "Use semantic search for paraphrases and concept-level retrieval.", "caveat": "Hybrid retrieval often combines their strengths."},
    {"topic": "accuracy vs F1-score", "a": "accuracy", "b": "F1-score", "summary": "Accuracy measures overall correctness, while F1 is the harmonic mean of precision and recall for a chosen positive class.", "choose_a": "Use accuracy when classes and error costs are reasonably balanced.", "choose_b": "Use F1 when positive-class precision and recall both matter under imbalance.", "caveat": "F1 hides threshold trade-offs and true-negative performance."},
    {"topic": "ROC-AUC vs PR-AUC", "a": "ROC-AUC", "b": "PR-AUC", "summary": "ROC-AUC summarizes true-positive versus false-positive rates, while PR-AUC summarizes precision versus recall.", "choose_a": "Use ROC-AUC for broad ranking discrimination across thresholds.", "choose_b": "Use PR-AUC when the positive class is rare and positive performance is central.", "caveat": "Neither metric selects an operating threshold."},
    {"topic": "standardization vs normalization", "a": "standardization", "b": "normalization", "summary": "Standardization uses mean and standard deviation, while normalization commonly rescales to a range or vector norm.", "choose_a": "Use standardization for many linear, distance, and gradient-based models.", "choose_b": "Use normalization when bounded ranges or unit-length vectors are required.", "caveat": "Document the exact definition because terminology varies."},
    {"topic": "batch inference vs online inference", "a": "batch inference", "b": "online inference", "summary": "Batch inference scores many records on a schedule, while online inference responds to individual requests with low latency.", "choose_a": "Use batch inference for periodic reports and large offline workloads.", "choose_b": "Use online inference for interactive or real-time decisions.", "caveat": "Freshness, cost, latency, and failure handling determine the architecture."},
    {"topic": "data drift vs concept drift", "a": "data drift", "b": "concept drift", "summary": "Data drift changes the input distribution, while concept drift changes the relationship between inputs and outcomes.", "choose_a": "Monitor data drift when incoming feature patterns change.", "choose_b": "Monitor concept drift through labeled performance when the predictive relationship changes.", "caveat": "Input drift can occur without performance loss, and concept drift can occur without obvious input drift."},
    {"topic": "greedy decoding vs beam search", "a": "greedy decoding", "b": "beam search", "summary": "Greedy decoding chooses the best next token at each step, while beam search keeps multiple candidate sequences.", "choose_a": "Use greedy decoding for speed and simple deterministic output.", "choose_b": "Use beam search when sequence-level quality benefits from broader search.", "caveat": "More beams increase latency and do not guarantee factuality."},
    {"topic": "BERTScore vs ROUGE", "a": "BERTScore", "b": "ROUGE", "summary": "BERTScore compares contextual token embeddings, while ROUGE measures lexical n-gram or sequence overlap.", "choose_a": "Use BERTScore for semantic similarity that tolerates paraphrasing.", "choose_b": "Use ROUGE for transparent overlap with reference wording.", "caveat": "Neither metric is a complete factuality or usefulness measure."},
    {"topic": "Gradio inference vs static browser inference", "a": "Gradio server inference", "b": "static browser inference", "summary": "Gradio commonly runs Python inference on hosted compute, while a static Transformers.js app runs ONNX inference in the visitor's browser.", "choose_a": "Use Gradio for Python-native models and flexible server-side logic.", "choose_b": "Use static inference for free hosting without a persistent backend.", "caveat": "Browser memory, model download size, and device support constrain static deployment."},
]

METRICS: list[dict[str, str]] = [
    {"topic": "accuracy", "definition": "Accuracy is the fraction of predictions that are correct.", "formula": "correct predictions divided by all predictions", "use": "Use it when classes and error costs are reasonably balanced.", "limitation": "It can be misleading for rare positive classes."},
    {"topic": "precision", "definition": "Precision is the fraction of predicted positives that are truly positive.", "formula": "TP divided by TP plus FP", "use": "Use it when false alarms are costly.", "limitation": "High precision can coexist with low recall."},
    {"topic": "recall", "definition": "Recall is the fraction of actual positives that the model finds.", "formula": "TP divided by TP plus FN", "use": "Use it when missed positives are costly.", "limitation": "High recall can create many false alarms."},
    {"topic": "F1-score", "definition": "F1-score is the harmonic mean of precision and recall.", "formula": "2 times precision times recall divided by precision plus recall", "use": "Use it when both positive-class precision and recall matter.", "limitation": "It ignores true negatives and depends on the decision threshold."},
    {"topic": "specificity", "definition": "Specificity is the fraction of actual negatives correctly identified.", "formula": "TN divided by TN plus FP", "use": "Use it to understand false-alarm control.", "limitation": "It does not describe positive-class detection."},
    {"topic": "balanced accuracy", "definition": "Balanced accuracy averages recall across classes, often sensitivity and specificity in binary classification.", "formula": "mean class recall", "use": "Use it when class imbalance makes ordinary accuracy misleading.", "limitation": "It still summarizes multiple error types into one number."},
    {"topic": "ROC-AUC", "definition": "ROC-AUC measures ranking discrimination across thresholds using true-positive and false-positive rates.", "formula": "area under the ROC curve", "use": "Use it to compare ranking quality across many cutoffs.", "limitation": "It can look optimistic when positives are extremely rare."},
    {"topic": "PR-AUC", "definition": "PR-AUC summarizes precision-recall performance across thresholds.", "formula": "area under the precision-recall curve", "use": "Use it when the positive class is rare and important.", "limitation": "Its baseline depends on positive prevalence."},
    {"topic": "log loss", "definition": "Log loss penalizes incorrect class probabilities, especially confident wrong predictions.", "formula": "negative mean log probability assigned to the true class", "use": "Use it when probability quality matters.", "limitation": "It is less intuitive than count-based metrics and sensitive to extreme probabilities."},
    {"topic": "Brier score", "definition": "Brier score is the mean squared error between predicted probabilities and binary outcomes.", "formula": "mean of probability minus outcome squared", "use": "Use it to assess probability accuracy and calibration.", "limitation": "It mixes calibration and discrimination effects."},
    {"topic": "MAE", "definition": "Mean Absolute Error averages absolute prediction errors.", "formula": "mean absolute value of prediction minus target", "use": "Use it for an interpretable average error in target units.", "limitation": "It treats all error sizes linearly."},
    {"topic": "MSE", "definition": "Mean Squared Error averages squared prediction errors.", "formula": "mean of prediction minus target squared", "use": "Use it when larger errors should receive greater penalty.", "limitation": "Its unit is squared and outliers can dominate."},
    {"topic": "RMSE", "definition": "Root Mean Squared Error is the square root of MSE.", "formula": "square root of mean squared error", "use": "Use it to retain target units while emphasizing large errors.", "limitation": "It is sensitive to outliers."},
    {"topic": "R-squared", "definition": "R-squared compares model squared error with a mean-prediction baseline.", "formula": "1 minus residual sum of squares divided by total sum of squares", "use": "Use it as a relative variance-explanation summary.", "limitation": "A high value does not prove good calibration, causality, or acceptable errors."},
    {"topic": "MAPE", "definition": "Mean Absolute Percentage Error averages absolute percentage errors.", "formula": "mean absolute error divided by actual value", "use": "Use it when percentage error is meaningful and targets stay away from zero.", "limitation": "It is undefined or unstable near zero and can be asymmetric."},
    {"topic": "silhouette score", "definition": "Silhouette score compares within-cluster cohesion with separation from other clusters.", "formula": "difference between nearest-cluster and own-cluster distance divided by the larger value", "use": "Use it as one diagnostic for clustering structure.", "limitation": "It favors some cluster shapes and does not replace domain validation."},
    {"topic": "BERTScore", "definition": "BERTScore aligns candidate and reference tokens using contextual embedding similarity.", "formula": "embedding-based precision, recall, and F1", "use": "Use it to measure semantic similarity when wording may differ.", "limitation": "It does not guarantee factual correctness or task completion."},
    {"topic": "ROUGE-L", "definition": "ROUGE-L measures overlap through the longest common subsequence between generated and reference text.", "formula": "precision, recall, or F-score from longest common subsequence", "use": "Use it as a lexical sequence-overlap metric for generated text.", "limitation": "It penalizes valid paraphrases and does not measure factuality."},
    {"topic": "perplexity", "definition": "Perplexity is the exponential of average token-level negative log likelihood.", "formula": "exp of average loss", "use": "Use it to summarize how well a language model predicts held-out target tokens.", "limitation": "It is not directly comparable across tokenizers and does not measure usefulness alone."},
    {"topic": "inference latency", "definition": "Inference latency is the elapsed time needed to produce a prediction or generated response.", "formula": "end time minus start time, usually summarized after warm-up", "use": "Use it to assess user-facing responsiveness and hardware trade-offs.", "limitation": "It depends on device, batch size, sequence length, caching, and decoding settings."},
]

WORKFLOWS: list[dict[str, Any]] = [
    {"topic": "classification project", "instruction": "Explain a practical end-to-end workflow for an imbalanced classification project.", "steps": ["define the decision, positive class, and error costs", "audit labels, duplicates, leakage, and time availability", "create grouped or time-aware train, validation, and test splits", "build a simple reproducible baseline", "fit preprocessing only on training data", "tune models with suitable cross-validation", "compare PR-AUC, recall, precision, calibration, and threshold outcomes", "review subgroup errors and representative false positives and false negatives", "package inference and monitor drift, latency, and outcome quality"], "pitfall": "Do not select the final threshold on the test set."},
    {"topic": "regression project", "instruction": "Explain a practical workflow for a regression project.", "steps": ["define target timing and business error cost", "inspect target distribution, missingness, and leakage", "choose a representative split", "build mean and linear baselines", "create a reproducible preprocessing pipeline", "compare MAE, RMSE, residuals, and subgroup behavior", "tune only inside the training-validation process", "test once and document uncertainty", "deploy with input validation and drift monitoring"], "pitfall": "A single R-squared value is not enough to judge model usefulness."},
    {"topic": "text classification", "instruction": "Explain a practical workflow for a text-classification project.", "steps": ["define labels and annotation guidelines", "review label agreement and leakage from metadata", "split by source, time, or entity when needed", "build keyword and TF-IDF baselines", "fine-tune a pretrained encoder model", "evaluate macro F1, per-class recall, calibration, and confusion patterns", "inspect short, long, ambiguous, and out-of-domain texts", "package tokenizer and model versions together", "monitor vocabulary and label drift"], "pitfall": "Near-duplicate texts must not appear across splits."},
    {"topic": "computer vision classification", "instruction": "Explain a practical workflow for an image-classification project.", "steps": ["define labels and image-quality criteria", "split by subject, device, batch, or site to prevent leakage", "inspect class balance and resolution", "build a pretrained CNN baseline", "apply realistic augmentation only to training images", "evaluate macro F1, confusion matrix, and subgroup performance", "review saliency or attention maps cautiously", "benchmark latency and model size", "monitor camera and environment drift"], "pitfall": "Random image-level splits can leak nearly identical images from the same source."},
    {"topic": "time-series forecasting", "instruction": "Explain a practical workflow for time-series forecasting.", "steps": ["define forecast horizon and decision cadence", "audit missing periods, revisions, and external variables", "create naive seasonal baselines", "use rolling-origin validation", "engineer lag and calendar features without future leakage", "compare MAE, RMSE, and horizon-specific errors", "inspect residual autocorrelation and seasonal failures", "retrain and monitor on a schedule"], "pitfall": "Random cross-validation is usually invalid for future forecasting."},
    {"topic": "anomaly detection", "instruction": "Explain a practical workflow for anomaly detection.", "steps": ["define what should trigger review", "separate known bad events from merely unusual events", "create robust features and scaling", "start with statistical and isolation-based baselines", "tune alert thresholds using review capacity", "measure precision at review budget and detection delay", "review false alerts with domain experts", "monitor distribution and alert-volume drift"], "pitfall": "An anomaly score is not automatically a defect probability."},
    {"topic": "RAG assistant", "instruction": "Explain a practical workflow for building a retrieval-augmented generation assistant.", "steps": ["define answer scope and source-of-truth documents", "clean, chunk, and version documents", "build keyword and embedding retrieval baselines", "evaluate recall at k on labeled questions", "rerank candidates when useful", "generate answers constrained to retrieved evidence", "require citations and test citation correctness", "measure groundedness, abstention, latency, and failure cases", "monitor document freshness and retrieval drift"], "pitfall": "A strong generator cannot compensate for consistently poor retrieval."},
    {"topic": "instruction tuning", "instruction": "Explain a practical workflow for instruction tuning a small domain model with LoRA.", "steps": ["define intended tasks and response style", "create and validate public-safe instruction-response data", "split by topic group to prevent paraphrase leakage", "measure the base model on held-out prompts", "configure LoRA and record trainable parameters", "train with validation loss and early stopping", "evaluate base and adapter using identical deterministic decoding", "perform semantic, lexical, latency, and human review", "publish the adapter, model card, and reproducible run artifacts"], "pitfall": "Do not claim instruction-tuning gains without a held-out base-versus-adapter comparison."},
    {"topic": "model deployment", "instruction": "Explain a practical workflow for deploying a machine-learning model.", "steps": ["freeze and version preprocessing, model, and dependencies", "define input and output schemas", "write unit, integration, and smoke tests", "benchmark latency and resource use", "choose batch, API, edge, or browser deployment", "add logging without exposing sensitive data", "monitor errors, drift, performance, and usage", "define rollback and retraining procedures"], "pitfall": "A notebook artifact alone is not a deployable inference contract."},
    {"topic": "model monitoring", "instruction": "Explain a practical workflow for monitoring a deployed model.", "steps": ["define service and model objectives", "log schema-valid inputs, predictions, confidence, latency, and errors", "track missingness and feature distributions", "join delayed labels when available", "measure performance and calibration by segment", "set actionable alerts and owners", "investigate drift with examples", "document retraining and rollback decisions"], "pitfall": "Monitoring without an operational response plan produces noise rather than reliability."},
    {"topic": "data quality", "instruction": "Explain a practical data-quality workflow before model training.", "steps": ["document source systems and field meanings", "profile completeness, uniqueness, validity, consistency, and timeliness", "trace duplicates and entity relationships", "check label definitions and temporal availability", "create automated validation rules", "separate correction from exclusion decisions", "version the cleaned dataset and validation report"], "pitfall": "Silently dropping problematic records can hide systematic bias."},
    {"topic": "feature store", "instruction": "Explain a practical workflow for building reusable ML features.", "steps": ["define feature owners and business meaning", "specify event time and availability time", "implement offline transformations", "ensure online and offline consistency", "version schemas and transformation code", "test freshness, null rates, and leakage", "monitor feature distributions and consumer usage"], "pitfall": "A feature computed with future information can invalidate every downstream model."},
    {"topic": "human evaluation of generated answers", "instruction": "Explain a practical workflow for human evaluation of an educational LLM.", "steps": ["define a rubric for correctness, relevance, instruction adherence, clarity, and safety", "sample prompts across categories and difficulty", "blind model identity when comparing systems", "use at least two reviewers for a representative subset", "record disagreements and adjudication", "report score distributions and examples", "connect manual findings to automated metrics"], "pitfall": "A single unblinded reviewer can introduce substantial subjective bias."},
    {"topic": "error analysis", "instruction": "Explain a practical workflow for model error analysis.", "steps": ["save per-example predictions and metadata", "sort examples by confidence and error severity", "group failures by class, segment, data source, and feature conditions", "review representative examples", "distinguish data, label, model, and threshold errors", "propose targeted experiments", "re-evaluate on an unchanged benchmark"], "pitfall": "Only reviewing the most dramatic errors can misrepresent the overall failure distribution."},
    {"topic": "reproducible experiment", "instruction": "Explain a practical workflow for a reproducible ML experiment.", "steps": ["freeze a dataset version and split IDs", "save configuration and random seeds", "record package, CUDA, and GPU versions", "store code commit and run ID", "save training logs, checkpoints, metrics, and predictions", "generate model and dataset cards", "provide one command or notebook to rerun the experiment"], "pitfall": "A random seed alone does not guarantee full GPU determinism."},
]

CODE_EXAMPLES: list[dict[str, str]] = [
    {"topic": "stratified train-test split", "instruction": "Generate a small Python example of a stratified train-test split.", "code": "from sklearn.model_selection import train_test_split\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.20, random_state=42, stratify=y\n)", "note": "Stratification preserves class proportions. Keep the test set untouched until final evaluation."},
    {"topic": "scikit-learn pipeline", "instruction": "Generate a small scikit-learn preprocessing and logistic-regression pipeline.", "code": "from sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\nmodel = Pipeline([\n    ('scale', StandardScaler()),\n    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))\n])\nmodel.fit(X_train, y_train)", "note": "A pipeline fits preprocessing only within the training workflow and reduces leakage risk."},
    {"topic": "cross-validation", "instruction": "Generate a small Python example of stratified cross-validation with F1 scoring.", "code": "from sklearn.model_selection import StratifiedKFold, cross_val_score\n\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nscores = cross_val_score(model, X, y, cv=cv, scoring='f1')\nprint(scores.mean(), scores.std())", "note": "Use grouped or time-aware splitters when observations are related."},
    {"topic": "confusion matrix", "instruction": "Generate a small Python example that calculates a confusion matrix and classification report.", "code": "from sklearn.metrics import confusion_matrix, classification_report\n\npred = model.predict(X_test)\nprint(confusion_matrix(y_test, pred))\nprint(classification_report(y_test, pred, digits=3))", "note": "Review per-class behavior instead of relying on accuracy alone."},
    {"topic": "threshold tuning", "instruction": "Generate a small Python example for evaluating classification thresholds.", "code": "import numpy as np\nfrom sklearn.metrics import precision_score, recall_score, f1_score\n\nproba = model.predict_proba(X_valid)[:, 1]\nfor threshold in np.arange(0.10, 0.91, 0.05):\n    pred = (proba >= threshold).astype(int)\n    print(threshold, precision_score(y_valid, pred), recall_score(y_valid, pred), f1_score(y_valid, pred))", "note": "Choose the threshold on validation data using error costs, then report once on test data."},
    {"topic": "text TF-IDF baseline", "instruction": "Generate a small Python TF-IDF text-classification baseline.", "code": "from sklearn.pipeline import Pipeline\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\n\ntext_model = Pipeline([\n    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2)),\n    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))\n])\ntext_model.fit(train_texts, train_labels)", "note": "A strong lexical baseline helps demonstrate the value of a Transformer fairly."},
    {"topic": "PyTorch GPU check", "instruction": "Generate a small Python example that checks whether PyTorch can use an NVIDIA GPU.", "code": "import torch\n\nprint('CUDA available:', torch.cuda.is_available())\nif torch.cuda.is_available():\n    print('GPU:', torch.cuda.get_device_name(0))\n    print('VRAM GB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))", "note": "The installed PyTorch build must match a supported CUDA runtime."},
    {"topic": "LoRA configuration", "instruction": "Generate a small PEFT LoRA configuration for a T5 sequence-to-sequence model.", "code": "from peft import LoraConfig, TaskType\n\nlora_config = LoraConfig(\n    task_type=TaskType.SEQ_2_SEQ_LM,\n    r=16,\n    lora_alpha=32,\n    lora_dropout=0.05,\n    target_modules=['q', 'v'],\n    bias='none'\n)", "note": "Confirm target module names against the selected architecture and report trainable parameters."},
    {"topic": "deterministic generation", "instruction": "Generate a small Transformers example for deterministic FLAN-T5 generation.", "code": "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM\n\nmodel_id = 'google/flan-t5-small'\ntokenizer = AutoTokenizer.from_pretrained(model_id)\nmodel = AutoModelForSeq2SeqLM.from_pretrained(model_id).to('cuda')\ninputs = tokenizer('Instruction: Explain precision.\\nResponse:', return_tensors='pt').to('cuda')\noutput = model.generate(**inputs, max_new_tokens=128, num_beams=4, do_sample=False)\nprint(tokenizer.decode(output[0], skip_special_tokens=True))", "note": "Use the same decoding configuration for base-versus-adapter evaluation."},
    {"topic": "JSON result saving", "instruction": "Generate a small Python example that saves model metrics to JSON.", "code": "import json\nfrom pathlib import Path\n\nmetrics = {'bertscore_f1': 0.0, 'latency_seconds': 0.0}\noutput = Path('outputs/model_metrics.json')\noutput.parent.mkdir(parents=True, exist_ok=True)\noutput.write_text(json.dumps(metrics, indent=2), encoding='utf-8')", "note": "Replace placeholder values only with metrics produced by an actual evaluation run."},
    {"topic": "pandas group summary", "instruction": "Generate a small pandas example that summarizes model metrics by prompt category.", "code": "summary = (\n    results.groupby(['model', 'category'])\n    .agg(bertscore_f1=('bertscore_f1', 'mean'), relevance=('semantic_relevance', 'mean'), examples=('id', 'count'))\n    .reset_index()\n)\nsummary.to_csv('outputs/category_metrics.csv', index=False)", "note": "Category-level results reveal whether gains are concentrated in only a few task types."},
    {"topic": "bootstrap confidence interval", "instruction": "Generate a small Python example for a bootstrap confidence interval of a paired metric improvement.", "code": "import numpy as np\n\nrng = np.random.default_rng(42)\ndelta = lora_scores - base_scores\nboot = [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(5000)]\nlow, high = np.percentile(boot, [2.5, 97.5])\nprint(delta.mean(), low, high)", "note": "A paired interval is more informative than comparing two unrelated aggregate means."},
]

QUALITY_ANALYTICS: list[dict[str, str]] = [
    {"topic": "quality case prioritization", "instruction": "Explain how machine learning could support quality case prioritization without replacing human review.", "output": "A supervised model can estimate review priority from fields available when a case arrives, such as severity, product family, age, and text embeddings. The score should rank work rather than make an irreversible decision. Evaluate recall for high-impact cases, calibration, subgroup behavior, and reviewer workload. Keep a human override, record reasons, and monitor drift and missed critical cases."},
    {"topic": "defect text classification", "instruction": "Explain how to build a model that classifies defect descriptions into issue categories.", "output": "Create clear annotation guidelines, review label agreement, remove duplicate or near-duplicate descriptions across splits, and build a TF-IDF baseline before fine-tuning a Transformer encoder. Report macro F1, per-class recall, confusion pairs, calibration, and examples of ambiguous language. Route low-confidence or novel cases to human review."},
    {"topic": "supplier risk", "instruction": "Explain a careful machine-learning approach to supplier quality risk scoring.", "output": "Define the decision and target period, use only information available before the prediction, and account for exposure such as shipment volume. Compare a transparent baseline with tree-based models, evaluate calibration and ranking at review capacity, and inspect performance by supplier size and product group. Treat the score as a screening aid because process changes and incomplete labels can create bias."},
    {"topic": "case volume forecasting", "instruction": "Explain how to forecast monthly quality case volume.", "output": "Start with naive and seasonal baselines, then use rolling-origin validation with lag, calendar, product-mix, and known operational features. Report MAE and RMSE by forecast horizon and product segment. Avoid using revised future totals or fields unavailable at forecast time, and monitor forecast bias after deployment."},
    {"topic": "root-cause recommendation", "instruction": "Explain why a root-cause recommendation model must be evaluated carefully.", "output": "Historical root-cause labels may reflect investigation practices rather than objective truth. Evaluate top-k recall, calibration, and performance by product and issue type, and review whether recommendations simply repeat common labels. Present ranked hypotheses with evidence rather than a definitive cause, require engineer confirmation, and monitor for feedback loops."},
    {"topic": "inspection image classifier", "instruction": "Explain a safe workflow for an inspection-image classifier.", "output": "Split images by physical unit, batch, device, or site to prevent near-duplicate leakage. Document lighting and resolution, use realistic augmentation, and compare a pretrained CNN with a Vision Transformer. Report per-class recall, confusion matrix, subgroup performance, latency, and representative failures. Low-confidence predictions should be reviewed rather than automatically accepted."},
    {"topic": "quality anomaly detection", "instruction": "Explain how anomaly detection can support quality analytics.", "output": "Build stable features from process measurements, use robust scaling, and compare statistical limits with isolation-based or reconstruction models. Tune the alert threshold to review capacity and measure precision among reviewed alerts, detection delay, and repeat-alert behavior. An anomaly is a review signal, not proof of a defect."},
    {"topic": "label discoloration trend", "instruction": "Give an example of using analytics to investigate an increase in label-discoloration cases.", "output": "Validate that the increase is not caused by reporting or exposure changes, normalize counts by production or shipment volume when possible, and break the trend down by supplier, lot, product family, site, and time. Use control charts or change-point diagnostics, then review representative cases with process experts. A predictive model should follow, not replace, the initial data-quality and root-cause investigation."},
    {"topic": "complaint duplicate detection", "instruction": "Explain how semantic similarity could help identify duplicate quality complaints.", "output": "Encode complaint summaries with a sentence Transformer, retrieve nearest candidates, and rerank them with a cross-encoder. Evaluate recall at k on reviewer-labeled duplicate pairs, false merges, latency, and performance on short or templated descriptions. Keep the final merge decision with a reviewer and preserve source records for auditability."},
    {"topic": "quality dashboard model metrics", "instruction": "Explain which model metrics should appear on a quality-analytics dashboard.", "output": "Show the decision threshold, sample size, prevalence, precision, recall, false-positive and false-negative counts, calibration, and latency. Add trends over time and breakdowns by product, site, and other important groups. Display data freshness and drift indicators, and link metrics to a review workflow rather than presenting a single accuracy number."},
    {"topic": "quality assistant grounding", "instruction": "Explain how to make an internal quality-learning assistant more trustworthy.", "output": "Use retrieval over approved, versioned documents; return source citations; restrict answers to retrieved evidence; and require abstention when support is weak. Evaluate retrieval recall, citation correctness, groundedness, and human usefulness. Protect confidential content through access controls and logging policies rather than exposing it in a public demo."},
    {"topic": "quality model deployment review", "instruction": "Explain a deployment review checklist for a quality machine-learning model.", "output": "Confirm data ownership, schema, timing, privacy, and label definition; reproduce training from versioned code and data; review test metrics and subgroup errors; validate inference parity and latency; define human override and failure handling; document intended and prohibited uses; and assign owners for drift monitoring, retraining, and rollback."},
]

RESPONSIBLE_AI: list[dict[str, str]] = [
    {"topic": "model card", "instruction": "Explain the purpose of a model card.", "output": "A model card documents a model's architecture, training data, intended and prohibited uses, evaluation results, limitations, bias considerations, and deployment details. It helps reviewers understand what was actually built and prevents a model artifact from being presented without context."},
    {"topic": "dataset card", "instruction": "Explain the purpose of a dataset card.", "output": "A dataset card documents dataset origin, format, creation or collection, cleaning, splits, licensing, sensitive-data handling, limitations, and appropriate uses. It makes the data lifecycle visible and supports reproducibility and risk review."},
    {"topic": "human oversight", "instruction": "Explain why human oversight is important for an educational LLM.", "output": "An educational LLM can produce fluent but incomplete or incorrect explanations and code. Human oversight is needed to verify factual claims, run code safely, resolve ambiguity, and decide whether an answer is suitable for a real task. The interface should show limitations and make review easy."},
    {"topic": "hallucination", "instruction": "Explain hallucination in a language model and how to evaluate it.", "output": "A hallucination is generated content that is unsupported, fabricated, or incorrect despite sounding plausible. Evaluation should combine reference-supported checks, targeted prompts, manual review of definitions and numeric claims, citation verification for grounded systems, and reporting of representative failures. No single automated metric proves factuality."},
    {"topic": "bias evaluation", "instruction": "Explain a practical approach to bias evaluation for a machine-learning model.", "output": "Identify groups and harms relevant to the use case, verify that group labels are appropriate and lawful to use, report performance and error rates by group with uncertainty, inspect data coverage and label quality, and test mitigation effects on both protected and overall performance. Document unresolved risks and human review requirements."},
    {"topic": "privacy", "instruction": "Explain privacy considerations for a public machine-learning demo.", "output": "Use synthetic or redistributable data, avoid logging sensitive prompts, display a warning not to submit private information, minimize stored data, and document third-party model and hosting behavior. Public demos should not contain confidential company cases, personal identifiers, or proprietary documents."},
    {"topic": "uncertainty communication", "instruction": "Explain how an ML assistant should communicate uncertainty.", "output": "The assistant should distinguish established definitions from context-dependent advice, state assumptions, avoid absolute claims, note when evidence is missing, and recommend verification against trusted documentation or experiments. Confidence language should reflect evidence rather than fluency."},
    {"topic": "out-of-scope requests", "instruction": "Explain how a domain ML assistant should handle out-of-scope requests.", "output": "The assistant should identify that the request is outside its ML/Data Science educational scope, avoid pretending to be an expert, and redirect the user to an appropriate source. It should still answer safe in-scope questions and should not over-refuse normal technical learning prompts."},
    {"topic": "evaluation transparency", "instruction": "Explain why evaluation transparency matters in a portfolio LLM project.", "output": "Evaluation transparency shows the exact dataset split, decoding settings, metric definitions, limitations, hardware, and per-example outputs behind aggregate numbers. It prevents invented or cherry-picked results and lets reviewers judge whether improvements are meaningful."},
    {"topic": "reproducible model release", "instruction": "Explain what should accompany a reproducible LoRA model release.", "output": "Publish the base-model identifier, adapter configuration and weights, tokenizer reference, dataset and split fingerprint, training configuration, package versions, evaluation outputs, model card, license notes, and example inference code. State clearly whether the model is an adapter, merged model, quantized model, or base-model demo."},
]


def _clean_inline(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\u00a0", " ")).strip()


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _record(
    instruction: str,
    output: str,
    category: str,
    topic: str,
    difficulty: str = "intermediate",
    input_text: str = "",
    topic_group: str | None = None,
) -> dict[str, Any]:
    instruction = _clean_inline(instruction)
    topic = _clean_inline(topic)
    topic_group = _clean_inline(topic_group or topic).lower()
    return {
        "instruction": instruction,
        "input": _clean_inline(input_text),
        "output": str(output).strip(),
        "response": str(output).strip(),
        "category": _clean_inline(category),
        "difficulty": _clean_inline(difficulty),
        "topic": topic,
        "topic_group": topic_group,
        "source": SOURCE_LABEL,
        "reference_answer": str(output).strip(),
        "id": f"mlds-ext-{_stable_id(instruction + '|' + topic_group)}",
        "split": "",
    }


def _concept_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in CONCEPTS:
        topic = item["topic"]
        simple = f"{item['definition']} {item['why']} Example: {item['example']} {item['caveat']}"
        practical = f"Practical example: {item['example']} Why it matters: {item['why']} Key caution: {item['caveat']}"
        interview = (
            f"A strong interview answer is: {item['definition']} {item['why']} "
            f"For example, {item['example']} I would also mention that {item['caveat'].lower()}"
        )
        rows.extend([
            _record(f"Explain {topic} in simple terms.", simple, "concept_explanation", topic, "beginner", topic_group=topic),
            _record(f"Give a practical ML example of {topic}.", practical, "example_generation", topic, "intermediate", topic_group=topic),
            _record(f"Give an interview-style answer explaining {topic}.", interview, "interview_answer", topic, "intermediate", topic_group=topic),
        ])
    return rows


def _comparison_records() -> list[dict[str, Any]]:
    rows = []
    for item in COMPARISONS:
        output = f"{item['summary']} {item['choose_a']} {item['choose_b']} {item['caveat']}"
        rows.append(_record(
            f"Compare {item['a']} and {item['b']}.", output, "algorithm_comparison", item["topic"], "intermediate", topic_group=item["topic"]
        ))
    return rows


def _metric_records() -> list[dict[str, Any]]:
    rows = []
    for item in METRICS:
        base = f"{item['definition']} Formula: {item['formula']}. {item['use']} Limitation: {item['limitation']}"
        business = (
            f"In a practical evaluation, {item['definition'].lower()} Compute it as {item['formula']}. "
            f"{item['use']} Always remember: {item['limitation']}"
        )
        rows.extend([
            _record(f"Explain {item['topic']} as a model-evaluation metric.", base, "metric_explanation", item["topic"], "beginner", topic_group=item["topic"]),
            _record(f"Explain how to interpret {item['topic']} in a real ML project.", business, "metric_explanation", item["topic"], "intermediate", topic_group=item["topic"]),
        ])
    return rows


def _workflow_records() -> list[dict[str, Any]]:
    rows = []
    for item in WORKFLOWS:
        numbered = " ".join(f"{i}. {step.capitalize()}." for i, step in enumerate(item["steps"], start=1))
        output = f"Recommended workflow: {numbered} Important pitfall: {item['pitfall']}"
        rows.append(_record(item["instruction"], output, "workflow_explanation", item["topic"], "advanced", topic_group=item["topic"]))
    return rows


def _code_records() -> list[dict[str, Any]]:
    rows = []
    for item in CODE_EXAMPLES:
        output = f"```python\n{item['code']}\n```\n{item['note']}"
        rows.append(_record(item["instruction"], output, "code_example", item["topic"], "intermediate", topic_group=item["topic"]))
    return rows


def _direct_records(items: list[dict[str, str]], category: str, difficulty: str) -> list[dict[str, Any]]:
    return [
        _record(item["instruction"], item["output"], category, item["topic"], difficulty, topic_group=item["topic"])
        for item in items
    ]


def _load_original(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        output = raw.get("output", raw.get("reference_answer", ""))
        rows.append(_record(
            raw.get("instruction", ""),
            output,
            raw.get("category", "concept_explanation"),
            raw.get("topic", "general"),
            raw.get("difficulty", "intermediate"),
            raw.get("input", ""),
            topic_group=raw.get("topic_group", raw.get("topic", "general")),
        ))
    return rows


def _deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for row in rows:
        key = _clean_inline(row["instruction"]).lower() + "\n" + _clean_inline(row.get("input", "")).lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _assign_group_splits(rows: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    groups = sorted({row["topic_group"] for row in rows}, key=lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest())
    n = len(groups)
    n_test = max(1, round(n * 0.10))
    n_validation = max(1, round(n * 0.10))
    n_train = n - n_validation - n_test
    split_by_group: dict[str, str] = {}
    for index, group in enumerate(groups):
        split_by_group[group] = "train" if index < n_train else ("validation" if index < n_train + n_validation else "test")
    for row in rows:
        row["split"] = split_by_group[row["topic_group"]]
    return rows


def _validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    prompt_counter = Counter()
    split_groups: defaultdict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        prompt_key = _clean_inline(row.get("instruction", "")).lower() + "\n" + _clean_inline(row.get("input", "")).lower()
        prompt_counter[prompt_key] += 1
        split_groups[row["topic_group"]].add(row["split"])
        for field in ("instruction", "output", "category", "topic", "topic_group", "split"):
            if not row.get(field):
                issues.append({"row": index, "type": "missing_field", "field": field})
        word_count = len(re.findall(r"\b\w+\b", row.get("output", "")))
        if word_count < 15:
            issues.append({"row": index, "type": "short_output", "words": word_count})
        if word_count > 260:
            issues.append({"row": index, "type": "long_output", "words": word_count})
    for key, count in prompt_counter.items():
        if count > 1:
            issues.append({"type": "duplicate_prompt", "count": count, "prompt": key[:120]})
    for group, splits in split_groups.items():
        if len(splits) > 1:
            issues.append({"type": "topic_group_leakage", "topic_group": group, "splits": sorted(splits)})
    return {"valid": not issues, "row_count": len(rows), "issue_count": len(issues), "issues": issues}


def _statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_words = [len(re.findall(r"\b\w+\b", row["instruction"] + " " + row.get("input", ""))) for row in rows]
    output_words = [len(re.findall(r"\b\w+\b", row["output"])) for row in rows]
    return {
        "rows": len(rows),
        "topic_groups": len({row["topic_group"] for row in rows}),
        "unique_topics": len({row["topic"] for row in rows}),
        "categories": dict(sorted(Counter(row["category"] for row in rows).items())),
        "difficulties": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "average_prompt_words": round(sum(prompt_words) / len(prompt_words), 2),
        "average_response_words": round(sum(output_words) / len(output_words), 2),
        "min_response_words": min(output_words),
        "max_response_words": max(output_words),
        "source": SOURCE_LABEL,
        "split_policy": "topic-grouped deterministic 80/10/10 split",
    }


def build_dataset(source: Path, output: Path, evaluation_output: Path, stats_output: Path, validation_output: Path) -> dict[str, Any]:
    rows = []
    rows.extend(_load_original(source))
    rows.extend(_concept_records())
    rows.extend(_comparison_records())
    rows.extend(_metric_records())
    rows.extend(_workflow_records())
    rows.extend(_code_records())
    rows.extend(_direct_records(QUALITY_ANALYTICS, "quality_analytics", "advanced"))
    rows.extend(_direct_records(RESPONSIBLE_AI, "responsible_ai", "intermediate"))
    rows = _assign_group_splits(_deduplicate(rows))
    rows = sorted(rows, key=lambda row: (row["split"], row["category"], row["topic_group"], row["instruction"]))

    validation = _validate(rows)
    if not validation["valid"]:
        raise ValueError(f"Extended dataset validation failed: {validation['issues'][:10]}")
    stats = _statistics(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    eval_rows = [
        {
            "id": row["id"],
            "instruction": row["instruction"],
            "input": row.get("input", ""),
            "reference_answer": row["reference_answer"],
            "category": row["category"],
            "topic": row["topic"],
            "topic_group": row["topic_group"],
        }
        for row in rows
        if row["split"] == "test"
    ]
    with evaluation_output.open("w", encoding="utf-8") as handle:
        for row in eval_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats_output.parent.mkdir(parents=True, exist_ok=True)
    stats_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    validation_output.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return {"dataset": str(output), "evaluation": str(evaluation_output), "statistics": stats, "validation": validation}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation-output", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION)
    args = parser.parse_args()
    result = build_dataset(args.source, args.output, args.evaluation_output, args.stats_output, args.validation_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
