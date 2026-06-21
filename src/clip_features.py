"""OpenCLIP encoder for the §6.6 representation transfer study.

Wraps ``open_clip`` with on-disk caching (per dataset × backbone) so the
~2-minute encoding pass runs once per dataset+backbone and is reused across
every K-sweep that consumes CLIP features.

Public functions
----------------
* :func:`load_clip_features_for_dataset` — main entry-point; returns L2-
  normalized image features and integer labels
* :func:`extract_clip_features` — lower-level helper that bypasses the
  per-dataset cache name and writes the supplied target path
"""

from __future__ import annotations

import os
import warnings

import numpy as np

dimensions: dict[str, int] = {
    "ViT-B-32": 512,
    "ViT-L-14": 768,
}
"""Feature dimensionality of every supported OpenCLIP backbone."""


def _get_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _preprocess_batch(batch_raw, preprocess):
    from PIL import Image

    pil_images = []
    for img in batch_raw:
        if img.ndim == 1:
            side = int(round(img.shape[0] ** 0.5))
            img = img.reshape(side, side)
        if img.ndim == 2:
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode="L").convert("RGB")
        elif img.ndim == 3 and img.shape[2] == 3:
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            pil_img = Image.fromarray(img_uint8, mode="RGB")
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")
        pil_images.append(preprocess(pil_img))
    return pil_images


def extract_clip_features(
    images_np: np.ndarray,
    y: np.ndarray,
    model_name: str,
    cache_path: str,
    batch_size: int = 128,
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode ``images_np`` with OpenCLIP ``model_name`` and return ``(X, y)``.

    Caches the (L2-normalized) embedding matrix to ``cache_path`` if it
    exists; otherwise runs a single GPU/MPS/CPU pass with batches of size
    ``batch_size``.
    """
    if not force_recompute and os.path.exists(cache_path):
        print(f"  Loading cached CLIP features from '{cache_path}'")
        data = np.load(cache_path)
        return data["X"], data["y"]

    import open_clip
    import torch

    device = _get_device()
    print(f"  Device: {device}")

    pretrained = "openai"  # the only public release we use

    print(f"  Loading {model_name} ({pretrained}) ...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="QuickGELU mismatch")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
    model = model.to(device)
    model.eval()

    n = len(images_np)
    print(f"  Encoding {n} images in batches of {batch_size} (preprocess + encode fused) ...")
    features_list = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_raw = images_np[start:start + batch_size]
            pil_images = _preprocess_batch(batch_raw, preprocess)
            batch_tensor = torch.stack(pil_images).to(device)
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features_list.append(feats.cpu().float().numpy())

            del batch_tensor, feats
            if device.type == "mps":
                torch.mps.empty_cache()

            done = min(start + batch_size, n)
            if done % (batch_size * 10) == 0 or done == n:
                print(f"    {done}/{n} images encoded ...", end="\r")

    print(f"    {n}/{n} images encoded.              ")

    X_clip = np.concatenate(features_list, axis=0).astype(np.float64)
    print(f"  Feature matrix shape: {X_clip.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez(cache_path, X=X_clip, y=y)
    print(f"  Cached to '{cache_path}'")

    del model, features_list
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return X_clip, y


def load_clip_features_for_dataset(
    dataset_name: str,
    images_np: np.ndarray,
    y: np.ndarray,
    model_name: str = "ViT-B-32",
    data_dir: str = "data",
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Public entry-point: dataset-named cache path under ``data_dir``."""
    safe_model = model_name.replace("/", "-").replace(" ", "_")
    cache_name = f"clip_{dataset_name.lower()}_{safe_model}.npz"
    cache_path = os.path.join(data_dir, cache_name)

    print(f"\n[CLIP] {dataset_name} | {model_name}")
    return extract_clip_features(
        images_np, y, model_name, cache_path, force_recompute=force_recompute
    )
