"""Build the self-authored ML/Data Science instruction dataset shipped with the project."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def _record(
    idx: int,
    instruction: str,
    output: str,
    category: str,
    topic: str,
    difficulty: str = "beginner",
    input_text: str = "",
) -> Dict[str, str]:
    split = "train" if idx % 10 < 8 else ("validation" if idx % 10 == 8 else "test")
    return {
        "id": f"ml_ds_{idx:04d}",
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "category": category,
        "difficulty": difficulty,
        "topic": topic,
        "source": "self-authored-synthetic",
        "split": split,
    }


CONCEPTS: Sequence[Tuple[str, str, str, str]] = [
    ("supervised learning", "Supervised learning trains a model on labeled examples where each input has a known target. The model learns a mapping from features to that target and is evaluated on unseen data.", "beginner", "machine learning fundamentals"),
    ("unsupervised learning", "Unsupervised learning works with unlabeled data to discover structure, such as clusters, lower-dimensional representations, or unusual observations. Results usually require domain interpretation.", "beginner", "machine learning fundamentals"),
    ("classification", "Classification predicts a discrete label, such as defect versus non-defect. Common models include logistic regression, decision trees, random forests, and neural networks.", "beginner", "supervised learning"),
    ("regression", "Regression predicts a continuous value, such as demand, cycle time, or temperature. Performance is commonly assessed with MAE, RMSE, or R-squared.", "beginner", "supervised learning"),
    ("clustering", "Clustering groups similar observations without known labels. K-means is common, but the number and meaning of clusters must be validated with metrics and domain review.", "beginner", "unsupervised learning"),
    ("feature engineering", "Feature engineering transforms raw data into useful model inputs. Examples include ratios, time-window aggregates, categorical encodings, interaction terms, and domain-specific indicators.", "beginner", "data preparation"),
    ("data leakage", "Data leakage occurs when training data contains information that would not be available at prediction time. It creates unrealistically strong validation results and poor production performance.", "intermediate", "model validation"),
    ("cross-validation", "Cross-validation repeatedly splits training data into folds so each fold is used for validation once. It provides a more stable performance estimate than a single split, especially on smaller datasets.", "intermediate", "model validation"),
    ("overfitting", "Overfitting happens when a model learns training noise or overly specific patterns and performs worse on unseen data. Regularization, simpler models, more data, and better validation can reduce it.", "beginner", "model validation"),
    ("underfitting", "Underfitting happens when a model is too simple or insufficiently trained to capture important patterns. It usually performs poorly on both training and validation data.", "beginner", "model validation"),
    ("regularization", "Regularization discourages overly complex models. L1 can drive some coefficients to zero, while L2 shrinks coefficients smoothly; both can improve generalization when tuned correctly.", "intermediate", "model validation"),
    ("class imbalance", "Class imbalance means one target class is much less frequent than another. Use stratified splits, suitable metrics, class weights, resampling, and threshold tuning rather than relying on accuracy alone.", "intermediate", "classification"),
    ("decision tree", "A decision tree splits data through a sequence of feature-based rules. It is easy to explain but can overfit unless depth, leaf size, or pruning is controlled.", "beginner", "algorithms"),
    ("random forest", "A random forest averages many decision trees trained on bootstrapped samples and random feature subsets. This usually improves stability and reduces overfitting compared with one tree.", "beginner", "algorithms"),
    ("gradient boosting", "Gradient boosting builds trees sequentially, with each new tree focusing on errors made by the current ensemble. It can be highly accurate but requires careful tuning and leakage control.", "intermediate", "algorithms"),
    ("logistic regression", "Logistic regression models the probability of a class using a linear combination of features passed through a logistic function. It is a strong interpretable baseline for binary classification.", "beginner", "algorithms"),
    ("principal component analysis", "PCA creates orthogonal components that capture decreasing amounts of variance. It can reduce dimensionality, but transformed components may be harder to interpret than original features.", "intermediate", "dimensionality reduction"),
    ("neural network", "A neural network applies layers of weighted transformations and nonlinear activation functions. Training adjusts the weights through backpropagation to reduce a chosen loss.", "beginner", "deep learning"),
    ("convolutional neural network", "A CNN uses learnable filters to detect local patterns such as edges, textures, and shapes. Weight sharing makes it effective for images and other grid-like data.", "beginner", "deep learning"),
    ("recurrent neural network", "An RNN processes a sequence step by step while carrying a hidden state. It can model order but may struggle with long-range dependencies and sequential computation costs.", "intermediate", "deep learning"),
    ("long short-term memory network", "An LSTM is an RNN variant with gates that control information flow. The gates help preserve useful signals over longer sequences and reduce vanishing-gradient problems.", "intermediate", "deep learning"),
    ("transformer", "A Transformer uses attention to model relationships between tokens in parallel. Self-attention lets each token weigh other tokens, making the architecture effective for language, vision, and multimodal tasks.", "intermediate", "transformers"),
    ("self-attention", "Self-attention computes query, key, and value representations so each token can combine information from other tokens. The attention weights indicate which relationships matter for the current representation.", "intermediate", "transformers"),
    ("transfer learning", "Transfer learning starts from a model pretrained on a broad dataset and adapts it to a new task. It often reduces data and compute requirements compared with training from scratch.", "beginner", "model development"),
    ("instruction tuning", "Instruction tuning trains a language model on instruction-and-response examples so it learns to follow natural-language tasks rather than only continue text.", "intermediate", "generative ai"),
    ("LoRA", "LoRA freezes the base model and learns small low-rank updates inside selected layers. It reduces trainable parameters and storage while preserving the reusable pretrained model.", "intermediate", "parameter-efficient fine-tuning"),
    ("PEFT", "Parameter-efficient fine-tuning adapts a pretrained model by training a small subset of parameters or added components. LoRA is one widely used PEFT method.", "intermediate", "parameter-efficient fine-tuning"),
    ("retrieval-augmented generation", "RAG retrieves relevant documents and provides them as context to a generator. This can improve grounding and freshness, but retrieval quality and citation design remain critical.", "intermediate", "generative ai"),
    ("embedding", "An embedding is a dense numerical representation in which semantically related items are placed near one another. Embeddings support search, clustering, recommendations, and retrieval.", "beginner", "representation learning"),
    ("model drift", "Model drift is a change in data patterns or the relationship between inputs and outcomes after deployment. Monitoring should compare feature distributions, performance, and business impact over time.", "intermediate", "mlops"),
    ("calibration", "A calibrated classifier produces probabilities that match observed frequencies. For example, predictions near 0.8 should be correct about 80 percent of the time across similar cases.", "advanced", "model evaluation"),
    ("SHAP values", "SHAP values estimate how each feature moves a prediction away from a baseline. They are useful for local and global interpretation but do not prove causality.", "advanced", "explainability"),
    ("data pipeline", "A data pipeline collects, validates, transforms, and delivers data for analytics or modeling. Reliable pipelines include tests, logging, lineage, and clear failure handling.", "beginner", "data engineering"),
    ("MLOps", "MLOps applies software and operations practices to machine-learning systems. It covers reproducible training, versioning, deployment, monitoring, governance, and controlled updates.", "intermediate", "mlops"),
    ("hallucination in language models", "A hallucination is a fluent but unsupported or incorrect model output. Mitigations include better data, retrieval grounding, constrained prompts, evaluation, and human review.", "intermediate", "responsible ai"),
]

METRICS: Sequence[Tuple[str, str]] = [
    ("accuracy", "Accuracy is the fraction of predictions that are correct. It is easy to understand but can be misleading when classes are imbalanced or error costs differ."),
    ("precision", "Precision is true positives divided by all predicted positives. Use it when false positives are costly, while also checking recall so the model does not miss too many real positives."),
    ("recall", "Recall is true positives divided by all actual positives. Use it when missing a positive case is costly, while reviewing precision to understand false alarms."),
    ("F1-score", "F1 is the harmonic mean of precision and recall. It balances both metrics but does not include true negatives and may hide whether precision or recall is the weaker component."),
    ("ROC-AUC", "ROC-AUC measures how well a classifier ranks positives above negatives across thresholds. It can look optimistic on highly imbalanced data, so precision-recall metrics may also be needed."),
    ("PR-AUC", "PR-AUC summarizes precision and recall across thresholds. It is especially informative when the positive class is rare and the cost of false positives matters."),
    ("MAE", "Mean absolute error averages the absolute prediction errors. It is expressed in the target's units and is less sensitive to large errors than RMSE."),
    ("RMSE", "Root mean squared error takes the square root of mean squared error. It is in the target's units and penalizes large errors more strongly than MAE."),
    ("R-squared", "R-squared describes the proportion of target variance explained relative to a mean baseline. It does not guarantee good predictions and can be negative on new data."),
    ("BERTScore", "BERTScore compares candidate and reference text using contextual token embeddings. It captures semantic similarity better than exact overlap, but it is not a direct test of factual correctness."),
]

COMPARISONS: Sequence[Tuple[str, str, str, str]] = [
    ("logistic regression", "decision tree", "Logistic regression is a linear probabilistic model with stable, interpretable coefficients. A decision tree learns nonlinear rules and interactions but can overfit. Start with logistic regression for a transparent baseline; use a tree when nonlinear thresholds are important.", "classification algorithms"),
    ("random forest", "gradient boosting", "Random forest builds many independent trees and averages them, making it robust and easier to tune. Gradient boosting builds trees sequentially to correct errors and can achieve higher accuracy, but it is more sensitive to hyperparameters and noise.", "ensemble learning"),
    ("precision", "recall", "Precision asks how many predicted positives are correct; recall asks how many real positives were found. Prioritize precision when false alarms are costly and recall when missed cases are costly.", "classification metrics"),
    ("MAE", "RMSE", "MAE weights every absolute error linearly and is easy to interpret. RMSE squares errors before averaging, so large misses receive more influence. Choose based on the business cost of large errors.", "regression metrics"),
    ("L1 regularization", "L2 regularization", "L1 uses absolute coefficient penalties and can create sparse models by setting some coefficients to zero. L2 uses squared penalties and usually shrinks all coefficients smoothly.", "regularization"),
    ("CNN", "Transformer", "CNNs emphasize local spatial patterns through convolution and strong image inductive biases. Vision Transformers use attention to model broader relationships and often benefit from larger pretraining datasets.", "deep learning architectures"),
    ("RNN", "Transformer", "RNNs process tokens sequentially through a recurrent state, while Transformers use attention and parallel processing. Transformers generally scale better and capture long-range relationships more directly.", "sequence modeling"),
    ("fine-tuning", "prompt engineering", "Fine-tuning changes model parameters using task examples and can create consistent domain behavior. Prompt engineering changes only the input instructions, making it faster and cheaper but often less stable across varied requests.", "language model adaptation"),
    ("full fine-tuning", "LoRA", "Full fine-tuning updates all model parameters and needs more memory and storage. LoRA freezes the base model and trains small adapters, making experimentation and deployment more practical on limited hardware.", "parameter-efficient fine-tuning"),
    ("clustering", "classification", "Clustering discovers groups without target labels, while classification learns to predict known labels from labeled examples. Cluster quality requires interpretation; classification quality can be measured against ground truth.", "learning paradigms"),
]

WORKFLOWS: Sequence[Tuple[str, str, str]] = [
    ("Build a binary classification project", "Define the decision and positive class, collect representative labeled data, split without leakage, build a simple baseline, engineer features, compare models with suitable metrics, tune the decision threshold, perform error analysis, document limitations, and monitor after deployment.", "ML project guidance"),
    ("Create a train-validation-test split", "Reserve the test set before model development. Use the training set to fit models, the validation set or cross-validation to select features and hyperparameters, and the test set once for the final unbiased estimate. Use time-based or group-based splitting when random splitting would leak information.", "Data Science workflow"),
    ("Handle missing values", "Measure missingness by column and segment, investigate why values are missing, choose deletion or imputation based on mechanism and impact, fit imputers only on training data, add missing indicators when useful, and verify that production preprocessing matches training.", "Data Science workflow"),
    ("Tune a classification threshold", "Generate validation probabilities, define the business cost of false positives and false negatives, inspect precision-recall tradeoffs, choose a threshold that meets the operational objective, and confirm performance on the untouched test set.", "ML project guidance"),
    ("Perform model error analysis", "Create a table of false positives and false negatives, segment errors by meaningful features, inspect difficult or mislabeled cases, compare errors across models, identify actionable causes, and convert findings into data, feature, or process improvements.", "Data Science workflow"),
    ("Monitor a deployed model", "Track input data quality, feature drift, prediction distributions, latency, failures, calibrated confidence, delayed ground-truth performance, and business outcomes. Define alert thresholds and an owner for investigation and retraining decisions.", "ML project guidance"),
    ("Design an instruction-tuning experiment", "Define supported tasks, create high-quality instruction-response records, split by topic, establish a base-model baseline, configure LoRA, train with reproducible settings, evaluate adherence, relevance, BERTScore where references exist, hallucinations, latency, and document limitations.", "ML project guidance"),
    ("Prepare data for time-series forecasting", "Sort by time, define the forecast horizon, create lag and rolling features using past information only, split chronologically, compare against naive baselines, use time-series cross-validation, and evaluate both average error and performance during important periods.", "Data Science workflow"),
]

CODE_EXAMPLES: Sequence[Tuple[str, str, str]] = [
    ("Show a small train-test split example in scikit-learn.", "```python\nfrom sklearn.model_selection import train_test_split\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y\n)\n```\n`stratify=y` preserves the class proportions in both splits for classification.", "data splitting"),
    ("Show a minimal logistic regression pipeline.", "```python\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\nmodel = make_pipeline(\n    StandardScaler(),\n    LogisticRegression(max_iter=1000, random_state=42),\n)\nmodel.fit(X_train, y_train)\n```\nThe pipeline fits scaling only on training data and applies the same transformation during inference.", "classification"),
    ("Show how to calculate precision and recall.", "```python\nfrom sklearn.metrics import precision_score, recall_score\n\nprecision = precision_score(y_true, y_pred, zero_division=0)\nrecall = recall_score(y_true, y_pred, zero_division=0)\n```\nReport both values and specify which class is treated as positive.", "classification metrics"),
    ("Show a simple cross-validation example.", "```python\nfrom sklearn.model_selection import StratifiedKFold, cross_val_score\n\ncv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\nscores = cross_val_score(model, X, y, cv=cv, scoring=\"f1\")\nprint(scores.mean(), scores.std())\n```\nThe mean estimates performance and the standard deviation shows fold-to-fold variability.", "model validation"),
    ("Show a small confusion matrix example.", "```python\nfrom sklearn.metrics import ConfusionMatrixDisplay\n\nConfusionMatrixDisplay.from_predictions(y_test, y_pred)\n```\nThe matrix separates true positives, true negatives, false positives, and false negatives.", "classification metrics"),
    ("Show how to save a scikit-learn pipeline.", "```python\nimport joblib\n\njoblib.dump(model, \"model.joblib\")\nloaded_model = joblib.load(\"model.joblib\")\n```\nSave the complete preprocessing-and-model pipeline so inference uses the same transformations.", "model deployment"),
]

QUALITY_ANALYTICS: Sequence[Tuple[str, str]] = [
    ("Explain how classification can support quality analytics.", "A classification model can estimate whether a new case is likely to require escalation. Features might include product family, issue type, quantity, supplier, and recent history. The model should support prioritization, not replace engineering review, and must be checked for leakage and class imbalance."),
    ("Give a quality analytics example for precision versus recall.", "For a model that flags potentially severe quality cases, high recall captures most truly severe cases but may create more false alarms. High precision reduces unnecessary investigations but can miss severe cases. The right threshold depends on investigation capacity and the cost of a missed issue."),
    ("How could anomaly detection be used in manufacturing quality?", "Anomaly detection can flag unusual sensor patterns, defect-rate shifts, or combinations of process variables that differ from normal production. Alerts should be reviewed with engineers because unusual does not automatically mean defective."),
    ("Describe a safe ML approach for root-cause prioritization.", "Use historical, non-confidential features to rank likely root-cause categories, keep a human engineer as the final decision-maker, show confidence and explanations, measure top-k accuracy, monitor drift, and avoid presenting model suggestions as confirmed causes."),
    ("How can a model support quality case triage?", "A triage model can rank incoming cases by expected severity, recurrence, or investigation urgency. It should be trained on clearly defined historical outcomes, evaluated for false negatives, integrated with operational capacity, and monitored for changes in products or reporting behavior."),
    ("Explain why time-based validation matters for quality data.", "Quality processes, suppliers, products, and inspection rules can change over time. A chronological split better estimates how a model trained on earlier cases will perform on future cases and helps reveal drift that random splitting may hide."),
]

INTERVIEW: Sequence[Tuple[str, str, str]] = [
    ("What is the bias-variance tradeoff?", "Bias is error from overly simple assumptions, while variance is sensitivity to training-data fluctuations. Increasing model complexity often lowers bias but raises variance. Good validation and regularization seek a balance that generalizes well.", "model validation"),
    ("Why should preprocessing be inside a pipeline?", "A pipeline ensures the same transformations are learned from training data and reused during validation and inference. It reduces leakage, simplifies deployment, and makes the modeling workflow reproducible.", "data preparation"),
    ("How do you choose an evaluation metric?", "Start from the decision and error costs. Consider target balance, whether ranking or calibrated probability matters, and operational constraints. Use multiple metrics when one number cannot represent all important tradeoffs.", "model evaluation"),
    ("What is data leakage and how do you prevent it?", "Data leakage is using information during training that would not be available at prediction time. Prevent it with time-aware splits, group-aware splits, training-only preprocessing, careful feature audits, and end-to-end pipelines.", "model validation"),
    ("What is the difference between a parameter and a hyperparameter?", "Parameters are learned from data, such as model weights. Hyperparameters are chosen before or around training, such as tree depth or learning rate, and are selected using validation rather than the final test set.", "machine learning fundamentals"),
    ("Why compare against a baseline?", "A baseline shows whether added complexity creates real value. It provides a minimum performance reference, helps detect leakage or implementation problems, and makes improvement claims easier to justify.", "model development"),
]


def build_dataset() -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    idx = 0
    for concept, explanation, difficulty, topic in CONCEPTS:
        records.append(_record(idx, f"Explain {concept} in simple terms.", explanation, "Concept explanation", topic, difficulty))
        idx += 1
    for metric, explanation in METRICS:
        records.append(_record(idx, f"Explain {metric} and mention one limitation.", explanation, "Metric explanation", "model evaluation", "intermediate"))
        idx += 1
    for left, right, answer, topic in COMPARISONS:
        records.append(_record(idx, f"Compare {left} and {right}.", answer, "Algorithm comparison", topic, "intermediate"))
        idx += 1
    for instruction, answer, category in WORKFLOWS:
        records.append(_record(idx, instruction + ".", answer, category, "end-to-end workflow", "intermediate"))
        idx += 1
    for instruction, answer, topic in CODE_EXAMPLES:
        records.append(_record(idx, instruction, answer, "Small code example", topic, "intermediate"))
        idx += 1
    for instruction, answer in QUALITY_ANALYTICS:
        records.append(_record(idx, instruction, answer, "Quality analytics", "quality analytics", "intermediate"))
        idx += 1
    for instruction, answer, topic in INTERVIEW:
        records.append(_record(idx, instruction, answer, "Interview-style answer", topic, "intermediate"))
        idx += 1

    beginner_prompts = [
        ("Explain a confusion matrix using a simple example.", "Imagine 100 inspected parts: 10 are actually defective. If the model correctly flags 8 defective parts, misses 2, incorrectly flags 5 good parts, and correctly clears 85 good parts, those four counts form the confusion matrix."),
        ("Explain one-hot encoding in beginner-friendly language.", "One-hot encoding turns each category into its own yes-or-no column. For colors red, blue, and green, a red item becomes red=1, blue=0, green=0. It avoids implying an artificial numeric order."),
        ("Explain why scaling can matter for machine learning.", "Scaling puts numeric features on comparable ranges. It is important for distance-based and gradient-based models because a large-unit feature can otherwise dominate. Tree models usually need it less."),
        ("Explain a validation set in simple terms.", "A validation set is a practice exam for the model. You train on one portion of data, use validation data to choose settings, and keep the test set untouched for the final evaluation."),
        ("Explain embeddings with an everyday analogy.", "Think of embeddings as coordinates on a meaning map. Items with similar meanings receive nearby coordinates, so a search system can find related content even when the exact words differ."),
        ("Explain model deployment in simple terms.", "Model deployment means making a trained model available to real users or systems. It also includes packaging preprocessing, handling inputs safely, logging failures, and monitoring performance after release."),
    ]
    for instruction, answer in beginner_prompts:
        records.append(_record(idx, instruction, answer, "Beginner-friendly explanation", "machine learning fundamentals", "beginner"))
        idx += 1

    examples = [
        ("Generate a small example of a binary classification problem.", "A factory wants to predict whether a part will pass inspection. Inputs include temperature, pressure, supplier, and machine ID. The target is pass=0 or fail=1. A model estimates failure probability so inspectors can prioritize high-risk parts."),
        ("Generate a small example of a regression problem.", "Predict the number of minutes required to complete a service request using request type, complexity, staffing level, and time of day. The target is a continuous duration, so MAE and RMSE are reasonable starting metrics."),
        ("Generate a small example of clustering.", "Group customers using purchase frequency, average order value, and product mix. The clusters may represent occasional, loyal, and high-value customers, but business experts should validate whether those labels are meaningful."),
        ("Generate a small example of data leakage.", "A hospital readmission model includes a code entered only after the patient is readmitted. The feature makes validation accuracy look excellent, but it is unavailable when the original prediction must be made, so it is leakage."),
        ("Generate a small example of threshold tuning.", "A fraud model outputs probabilities. At threshold 0.5 it catches 70 percent of fraud with few false alerts; at 0.3 it catches 90 percent but doubles reviews. The team chooses a threshold based on fraud cost and analyst capacity."),
        ("Generate a small example of model drift.", "A demand model was trained before a new sales channel launched. After launch, customer behavior and feature distributions change, and forecast error rises. Drift monitoring detects the shift and triggers investigation."),
    ]
    for instruction, answer in examples:
        records.append(_record(idx, instruction, answer, "Example generation", "applied machine learning", "beginner"))
        idx += 1

    return records
