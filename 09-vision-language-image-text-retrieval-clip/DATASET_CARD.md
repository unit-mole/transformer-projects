# Dataset Card — Safe Synthetic CLIP Demo Gallery

## Dataset name

`clip-safe-synthetic-gallery-v1`

## Purpose

A small, redistributable image-caption gallery for demonstrating browser-based CLIP retrieval and zero-shot classification without committing private, restricted, or copyrighted datasets.

## Composition

- **Images:** 12 original synthetic PNG scenes
- **Captions:** 12 primary captions plus tags and categories
- **Captions per image:** one primary caption in the public demo
- **Splits:** demo gallery only; evaluation relevance judgments are stored separately
- **Image path field:** `image_path`
- **Caption field:** `caption`
- **Category field:** `category`

## Metadata fields

`image_id`, `image_path`, `caption`, `category`, `tags`, `source`, `license_note`, and `embedding_id`.

## Sources and license note

The PNG scenes are programmatically drawn original assets created for this project. The repository license applies to these assets. They do not contain real people, private information, brands, workplace material, or copied photographs.

## Cleaning and validation

- RGB conversion and valid-image checks
- fixed public-demo dimensions
- EXIF-free generated images
- duplicate ID/path validation
- safe caption and metadata review
- relative paths compatible with GitHub Pages

## Known limitations

The gallery is intentionally tiny and visually simple. It is useful for deployment validation and qualitative exploration, not for estimating general CLIP performance. A permitted Flickr8k subset may be used locally for stronger evaluation, but the full dataset is not redistributed here.

## Sensitive-data handling

No sensitive or personal data is included. Users should not add private photos, IDs, medical images, confidential inspection images, or copyrighted material without appropriate permission.

## Example record

```json
{
  "image_id": "img_001",
  "image_path": "./sample_images/dog_running.png",
  "caption": "A brown dog running across green grass under a blue sky.",
  "category": "animals",
  "tags": ["dog", "running", "grass", "outdoors"],
  "source": "original synthetic project asset",
  "license_note": "Repository license applies",
  "embedding_id": "emb_img_001"
}
```
