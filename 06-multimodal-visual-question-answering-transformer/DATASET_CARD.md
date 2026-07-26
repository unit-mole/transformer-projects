# Dataset Card — Project 06 VQA Samples

## Repository dataset

The committed dataset is a safe sample-only VQA-style dataset generated for this
project. It contains three synthetic images and three question-answer records.

- **Images:** 3
- **Questions:** 3
- **Splits:** `demo`
- **Question categories:** color, number, object
- **Sensitive data:** none
- **People:** none
- **Redistribution:** project-generated synthetic assets

## VQA v2 compatibility

The loader and evaluator support VQA v2-style fields such as `image_id`,
`image_path`, `question`, `answer`, multiple `answers`, `question_type`,
`answer_type`, and `split`. The full VQA v2 dataset is not redistributed.

## Cleaning and validation

Sample images are RGB PNG files. Paths are validated by
`scripts/prepare_sample_data.py`. Question preprocessing trims extra whitespace
and limits length without removing meaning-bearing terms.

## Limitations

Three synthetic examples are sufficient only for smoke testing and interface
validation. They are not a representative evaluation dataset and must not be
used to claim model accuracy.

## Responsible use

Do not add private photos, IDs, medical images, confidential workplace content,
sensitive personal information, or copyrighted images without authorization.
