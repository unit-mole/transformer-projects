# Data preparation

The committed public demo uses 12 original synthetic PNG images. The full Flickr8k dataset is not included. To use a permitted local Flickr8k subset, place images under `data/raw/images/`, create a captions CSV with `image_id,image_path,caption,category,tags`, and adapt `scripts/prepare_gallery.py`.

The browser consumes:

- `web/data/image_gallery.json`
- `web/data/captions.json`
- `web/data/image_embeddings.json`
- `web/data/retrieval_eval_queries.json`

Do not commit private, confidential, proprietary, medical, identity, or copyrighted images without permission.
