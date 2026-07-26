# Data documentation

The repository contains only three project-generated synthetic images and a
small VQA-style CSV for smoke tests and interface examples. The images do not
contain people or private information.

The project is compatible with VQA v2-style records containing an image,
question, multiple human answers, question type, and answer type. The full VQA
v2 dataset is intentionally excluded because it is large and has its own usage
terms. Download it from its official source only when running a documented
evaluation.

## Public sample schema

- `image_id`
- `image_path`
- `question`
- `answer`
- `answers` — JSON list of reference answers
- `question_type`
- `answer_type`
- `category`
- `split`
- `source`

Do not add personal photos, IDs, medical images, confidential workplace
images, copyrighted images without permission, or images containing sensitive
personal data.
