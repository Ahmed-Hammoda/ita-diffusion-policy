"""Pre-compute frozen DINOv2 image features and CLIP text features once.   [Owner: Data — week 2]

Why: training end-to-end through the image encoder does not fit on a 6 GB GPU. The encoders are
frozen anyway, so we run them ONCE over every frame and store small vectors on disk.

Plan:
  1. Load `lerobot/libero` with LeRobotDataset (see scripts/01_explore_dataset.py).
  2. For every frame and both cameras: create `n_aug_views` views
       view 0 = clean (resize to 224), views 1.. = random crop + colour jitter, then resize to 224,
     run DINOv2 (`facebook/dinov2-small`, use the CLS token / pooler output, 384-d) in fp16 on GPU.
  3. Store per episode: scene (T, V, 384), wrist (T, V, 384), state (T, 8), action (T, 7) as fp16 .npz
     → roughly 1–3 GB for the whole dataset.
  4. Encode each unique instruction with the CLIP text encoder
     (`openai/clip-vit-base-patch32`, CLIPTextModelWithProjection → text_embeds, 512-d) → lang.npz.
  5. Also store paraphrases (for the language-grounding test) with the same text encoder.

Use torch.no_grad() + model.eval(); batch 256 images at a time.
"""

from __future__ import annotations


def cache_image_features(cfg: dict) -> None:
    raise NotImplementedError("TODO (Data)")


def cache_text_features(cfg: dict, instructions: list[str]) -> None:
    raise NotImplementedError("TODO (Data)")
