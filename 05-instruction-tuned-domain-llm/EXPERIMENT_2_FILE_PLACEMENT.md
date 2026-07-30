# File placement

Extract the overlay so the folder structure becomes:

```text
05-instruction-tuned-domain-llm\
├── EXPERIMENT_2_README.md
├── EXPERIMENT_2_FILE_PLACEMENT.md
├── requirements-experiment2.txt
├── data\
│   ├── curated_topic_cards_v3.json
│   ├── curated_comparisons_v3.json
│   ├── curated_code_examples_v3.json
│   ├── curated_workflows_v3.json
│   └── experiment2_quality_rules.json
├── src\
│   ├── experiment_archive.py
│   ├── experiment2_dataset.py
│   ├── experiment2_training.py
│   └── experiment2_comparison.py
├── scripts\
│   ├── archive_experiment_1.py
│   ├── build_dataset_v3.py
│   └── compare_experiment_1_vs_2.py
├── notebooks\
│   └── 05_experiment_2_quality_upgrade_pipeline.ipynb
├── docs\
│   ├── EXPERIMENT_1_PRESERVATION_GUIDE.md
│   └── EXPERIMENT_2_GUIDE.md
└── tests\
    ├── test_experiment_archive.py
    ├── test_experiment2_dataset.py
    └── test_experiment2_comparison.py
```

Existing files and the full `outputs\experiments\flan_t5_base_lora_20260730_120312\`
folder remain in place.
