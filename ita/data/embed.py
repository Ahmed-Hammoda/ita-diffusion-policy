"""Pre-compute frozen DINOv2 image features and CLIP text features once.

Why: training end-to-end through the image encoder does not fit on a 6 GB GPU. The encoders are
frozen anyway, so we run them ONCE over every frame and store small vectors on disk.

Pipeline (run with scripts/05_build_cache.py):
  1. Read states, actions and task indices straight from the dataset's parquet files, and decode
     the two camera videos sequentially with torchcodec (much faster than LeRobotDataset's random
     access, and it needs no internet). Dataset images are upright; no flip here.
  2. For every frame and both cameras build `n_aug_views` views on the GPU:
        view 0  = clean: the full 256x256 image resized to 224
        views 1.. = random crop (side 85-100 %) + colour jitter (brightness, contrast, saturation, hue),
                    drawn once per (episode, camera, view) and applied to every frame of the episode,
                    so consecutive frames of a view differ only by real motion
     then run DINOv2 (`facebook/dinov2-small`, pooler output = layer-normed CLS token, 384-d) in fp16.
  3. Store one file per episode, `<cache_dir>/episodes/ep{NNNN}.npz`:
        scene (T, V, 384) fp16, wrist (T, V, 384) fp16, state (T, 8) f32, action (T, 7) f32,
        task_index (), episode_index ()
  4. Encode the 40 instructions with the CLIP text encoder (`CLIPTextModelWithProjection`,
     text_embeds, 512-d, L2-normalised) -> `<cache_dir>/lang.npz`. Paraphrases for the language
     tests (Step 15) use the same function with another `name`.
  5. `<cache_dir>/meta.json` records the encoders, augmentation settings and dataset statistics.

The build is resumable: episodes whose file already exists are skipped. Augmentations are seeded per
episode, so a rebuilt episode is identical.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# task_index blocks of `lerobot/libero` (see docs/data_conventions.md).
SUITE_TASKS = {
    "libero_10": range(0, 10),
    "libero_goal": range(10, 20),
    "libero_object": range(20, 30),
    "libero_spatial": range(30, 40),
}
CAMERAS = {"scene": "observation.images.image", "wrist": "observation.images.image2"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

AUG = {  # augmentation ranges for views 1..V-1
    "crop_side": (0.85, 1.0),    # crop side as a fraction of the image side (area 72-100 %)
    "brightness": 0.3,           # factor in [1 - 0.3, 1 + 0.3]
    "contrast": 0.3,
    "saturation": 0.3,
    "hue": 0.05,                 # rotation of the colour wheel by up to +-0.05 turn (18 degrees)
}


# --------------------------------------------------------------------------- paths and metadata

def cache_dir(cfg: dict) -> Path:
    return Path(os.path.expanduser(cfg["data"]["cache_dir"]))


def dataset_root(repo_id: str = "lerobot/libero") -> Path:
    """Local snapshot of a LeRobot dataset (downloaded earlier by LeRobotDataset / scripts/01)."""
    home = Path(os.environ.get("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot")))
    snaps = sorted((home / "hub" / f"datasets--{repo_id.replace('/', '--')}" / "snapshots").glob("*"))
    if not snaps:
        raise FileNotFoundError(f"{repo_id} not found under {home}/hub. Download it first "
                                f"(python scripts/01_explore_dataset.py downloads it).")
    return snaps[-1]


def load_tasks(root: Path) -> list[str]:
    """Instruction string of every task_index, in task_index order (40 strings)."""
    t = pd.read_parquet(root / "meta" / "tasks.parquet")
    return [s for s, _ in sorted(zip(t.index, t["task_index"]), key=lambda p: p[1])]


def load_episode_table(root: Path) -> pd.DataFrame:
    """One row per episode: length, video file and time range of each camera."""
    files = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    cols = ["episode_index", "length"] + [f"videos/{key}/{c}" for key in CAMERAS.values()
                                          for c in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    eps = pd.concat([pd.read_parquet(f, columns=cols) for f in files]).set_index("episode_index").sort_index()
    return eps


def load_lowdim(root: Path) -> dict[int, dict]:
    """episode_index -> {"state": (T, 8) f32, "action": (T, 7) f32, "task_index": int}."""
    files = sorted((root / "data").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "task_index",
                                                "observation.state", "action"]) for f in files])
    out = {}
    for ep, g in df.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
        assert (g["frame_index"].to_numpy() == np.arange(len(g))).all(), f"frames missing in episode {ep}"
        assert g["task_index"].nunique() == 1
        out[int(ep)] = {"state": np.stack(g["observation.state"].to_numpy()).astype(np.float32),
                        "action": np.stack(g["action"].to_numpy()).astype(np.float32),
                        "task_index": int(g["task_index"].iloc[0])}
    return out


def select_episodes(lowdim: dict[int, dict], suite: str = "all",
                    task_indices: list[int] | None = None) -> list[int]:
    """Episode ids of a suite ('all' or a SUITE_TASKS key), optionally only some DATASET task indices.

    To select simulator tasks, convert their ids first with `sim_to_dataset_tasks` (the dataset and the
    simulator order the tasks differently; never use a position inside the block).
    """
    tasks = set(range(40)) if suite == "all" else set(SUITE_TASKS[suite])
    if task_indices is not None:
        tasks &= set(task_indices)
    return sorted(ep for ep, d in lowdim.items() if d["task_index"] in tasks)


def sim_task_names(suite: str) -> list[str]:
    """Instructions of a LIBERO suite in SIMULATOR order (task id 0, 1, ...)."""
    from libero.libero import benchmark
    s = benchmark.get_benchmark_dict()[suite]()
    return [s.get_task(i).language for i in range(s.n_tasks)]


def sim_to_dataset_tasks(suite: str, sim_ids: list[int] | None, dataset_tasks: list[str]) -> list[int]:
    """Dataset task_index of each simulator task id of `suite`, matched by instruction string."""
    names = sim_task_names(suite)
    ids = range(len(names)) if sim_ids is None else sim_ids
    lookup = {t.strip().lower(): i for i, t in enumerate(dataset_tasks)}
    out = []
    for i in ids:
        key = names[i].strip().lower()
        if key not in lookup:
            raise KeyError(f"{suite} task {i} {names[i]!r} not found among the dataset instructions")
        out.append(lookup[key])
    return out


# --------------------------------------------------------------------------- video decoding

class VideoReader:
    """Decodes episode frames from LeRobot v3 video files (one mp4 holds many episodes)."""

    def __init__(self, root: Path, episodes: pd.DataFrame, fps: float = 10.0):
        self.root, self.eps, self.fps = root, episodes, fps
        self._decoders: dict[str, object] = {}

    def _decoder(self, path: str):
        if path not in self._decoders:
            from torchcodec.decoders import VideoDecoder
            if len(self._decoders) >= 4:                       # keep a few files open, not all 74
                self._decoders.pop(next(iter(self._decoders)))
            self._decoders[path] = VideoDecoder(path, dimension_order="NCHW")
        return self._decoders[path]

    def frames(self, ep: int, camera: str) -> torch.Tensor:
        """(T, 3, 256, 256) uint8, upright, for camera 'scene' or 'wrist'."""
        key, row = CAMERAS[camera], self.eps.loc[ep]
        path = str(self.root / "videos" / key / f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}"
                   / f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4")
        start = round(row[f"videos/{key}/from_timestamp"] * self.fps)
        T = int(row["length"])
        frames = self._decoder(path).get_frames_in_range(start, start + T).data
        assert frames.shape[0] == T, f"episode {ep} {camera}: decoded {frames.shape[0]} frames, expected {T}"
        return frames


# --------------------------------------------------------------------------- augmentation (GPU, batched)

def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)


def _hue_rotate(x: torch.Tensor, turns: torch.Tensor) -> torch.Tensor:
    """Rotate hue by `turns` (N,) of a full turn, in YIQ space (keeps brightness)."""
    to_yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]],
                          device=x.device, dtype=x.dtype)
    yiq = torch.einsum("ij,njhw->nihw", to_yiq, x)
    a = 2 * math.pi * turns
    c, s = torch.cos(a)[:, None, None], torch.sin(a)[:, None, None]
    i, q = yiq[:, 1], yiq[:, 2]
    yiq = torch.stack([yiq[:, 0], c * i - s * q, s * i + c * q], dim=1)
    return torch.einsum("ij,njhw->nihw", torch.linalg.inv(to_yiq), yiq)


def sample_aug_params(n_aug: int, device: str | torch.device = "cpu",
                      generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """Random parameters for `n_aug` augmented views, each a tensor of shape (n_aug,)."""
    U = lambda: torch.rand(n_aug, device=device, generator=generator)       # noqa: E731
    jit = lambda s: 1 - s + 2 * s * U()                                     # noqa: E731
    lo, hi = AUG["crop_side"]
    side = lo + (hi - lo) * U()
    return {"side": side,
            "tx": (1 - side) * (2 * U() - 1), "ty": (1 - side) * (2 * U() - 1),   # keeps the crop inside
            "brightness": jit(AUG["brightness"]), "contrast": jit(AUG["contrast"]),
            "saturation": jit(AUG["saturation"]), "hue": AUG["hue"] * (2 * U() - 1)}


def make_views(imgs: torch.Tensor, n_views: int, size: int = 224,
               params: dict[str, torch.Tensor] | None = None,
               generator: torch.Generator | None = None) -> torch.Tensor:
    """imgs (N, 3, H, W) uint8 on the GPU -> (N, n_views, 3, size, size) float in [0, 1].

    View 0 is the clean image; view v >= 1 gets a random crop + colour jitter with parameters
    `params[...][v - 1]`, the SAME for all N frames. Encoding an episode with one set of params keeps
    every view consistent over time, so the difference between history frames is real motion,
    not augmentation noise. If `params` is None they are drawn here (one set for this batch).
    """
    N = imgs.shape[0]
    x = imgs.float() / 255.0
    views = [F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False)]
    if n_views > 1:
        A = n_views - 1
        p = params if params is not None else sample_aug_params(A, x.device, generator)
        p = {k: v.to(x.device).repeat_interleave(N) for k, v in p.items()}     # (A*N,): view-major, like xa
        M = A * N
        xa = x.repeat(A, 1, 1, 1)                                             # (M, 3, H, W)

        # Crop via an affine sampling grid: scale = crop side, shift = crop position.
        theta = torch.zeros(M, 2, 3, device=x.device)
        theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = p["side"], p["side"], p["tx"], p["ty"]
        grid = F.affine_grid(theta, (M, 3, size, size), align_corners=False)
        xa = F.grid_sample(xa, grid, mode="bilinear", padding_mode="border", align_corners=False)

        # Colour jitter.
        col = lambda k: p[k][:, None, None, None]                              # noqa: E731
        xa = xa * col("brightness")
        mean = _rgb_to_gray(xa).mean(dim=(2, 3), keepdim=True)
        xa = (xa - mean) * col("contrast") + mean
        gray = _rgb_to_gray(xa)
        xa = (xa - gray) * col("saturation") + gray
        xa = _hue_rotate(xa.clamp(0, 1), p["hue"])
        views += list(xa.clamp(0, 1).reshape(A, N, 3, size, size))
    return torch.stack(views, dim=1)


# --------------------------------------------------------------------------- encoders

class DinoEncoder:
    """Frozen DINOv2: images in [0, 1] -> 384-d pooled CLS features."""

    def __init__(self, name: str = "facebook/dinov2-small", device: str | torch.device = "cuda"):
        from transformers import AutoModel
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(name).to(self.device, self.dtype).eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
        """x: (N, 3, 224, 224) in [0, 1] -> (N, 384) float32 on CPU."""
        out = []
        for i in range(0, x.shape[0], batch_size):
            b = ((x[i:i + batch_size].to(self.device) - self.mean) / self.std).to(self.dtype)
            out.append(self.model(pixel_values=b).pooler_output.float().cpu())
        return torch.cat(out)


class ClipTextEncoder:
    """Frozen CLIP text tower: instruction strings -> 512-d L2-normalised embeddings."""

    def __init__(self, name: str = "openai/clip-vit-base-patch32", device: str | torch.device = "cpu"):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()      # the checkpoint also holds the (unused) vision tower: silence that report
        self.device = torch.device(device)
        self.tok = CLIPTokenizer.from_pretrained(name)
        self.model = CLIPTextModelWithProjection.from_pretrained(name).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> torch.Tensor:
        t = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        e = self.model(**t).text_embeds.float()
        return F.normalize(e, dim=-1).cpu()


# --------------------------------------------------------------------------- caching

@torch.no_grad()
def encode_episode(reader: VideoReader, encoder: DinoEncoder, ep: int, n_views: int, image_size: int,
                   seed: int, frames: dict[str, torch.Tensor] | None = None,
                   frames_per_batch: int = 64) -> dict[str, np.ndarray]:
    """DINOv2 features of one episode: {"scene": (T, V, 384), "wrist": (T, V, 384)} fp16."""
    out = {}
    for c_i, camera in enumerate(CAMERAS):
        imgs = frames[camera] if frames is not None else reader.frames(ep, camera)
        # One set of augmentation parameters per (episode, camera, view), shared by all frames.
        g = torch.Generator(device=encoder.device).manual_seed(seed * 1_000_003 + ep * 2 + c_i)
        params = sample_aug_params(n_views - 1, encoder.device, g) if n_views > 1 else None
        feats = []
        for i in range(0, imgs.shape[0], frames_per_batch):
            v = make_views(imgs[i:i + frames_per_batch].to(encoder.device), n_views, image_size, params)  # (n, V, 3, S, S)
            n = v.shape[0]
            feats.append(encoder(v.flatten(0, 1)).view(n, n_views, -1))
        out[camera] = torch.cat(feats).half().numpy()
    return out


def cache_image_features(cfg: dict, episodes: list[int] | None = None, overwrite: bool = False,
                         device: str | None = None, log_every: int = 25) -> Path:
    """Build `<cache_dir>/episodes/epNNNN.npz` for the given episodes (default: all 1693)."""
    root = dataset_root(cfg["data"]["repo_id"])
    out_dir = cache_dir(cfg) / "episodes"
    out_dir.mkdir(parents=True, exist_ok=True)

    lowdim = load_lowdim(root)
    table = load_episode_table(root)
    episodes = sorted(lowdim) if episodes is None else list(episodes)
    todo = [ep for ep in episodes if overwrite or not (out_dir / f"ep{ep:04d}.npz").exists()]
    print(f"[cache] {len(episodes)} episodes requested, {len(episodes) - len(todo)} already cached, {len(todo)} to do")
    write_meta(cfg, root)
    if not todo:
        return out_dir

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DinoEncoder(cfg["encoders"]["vision"], device)
    n_views, size, seed = cfg["data"]["n_aug_views"], cfg["encoders"]["image_size"], cfg.get("seed", 0)

    # Decode the next episode on a background thread while the GPU encodes the current one.
    reader = VideoReader(root, table)
    decode = lambda ep: {cam: reader.frames(ep, cam) for cam in CAMERAS}      # noqa: E731
    pool = ThreadPoolExecutor(max_workers=1)
    nxt = pool.submit(decode, todo[0])
    t0, n_frames = time.time(), 0
    for i, ep in enumerate(todo):
        frames = nxt.result()
        if i + 1 < len(todo):
            nxt = pool.submit(decode, todo[i + 1])
        feats = encode_episode(reader, encoder, ep, n_views, size, seed, frames)
        d = lowdim[ep]
        tmp = out_dir / f"ep{ep:04d}.tmp.npz"
        np.savez(tmp, scene=feats["scene"], wrist=feats["wrist"], state=d["state"], action=d["action"],
                 task_index=np.int64(d["task_index"]), episode_index=np.int64(ep))
        tmp.replace(out_dir / f"ep{ep:04d}.npz")                                  # atomic: never half-written
        n_frames += len(d["action"])
        if (i + 1) % log_every == 0 or i + 1 == len(todo):
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(todo) - i - 1)
            print(f"[cache] {i + 1}/{len(todo)} episodes | {n_frames / dt:.0f} frames/s "
                  f"| elapsed {dt / 60:.1f} min | eta {eta / 60:.1f} min", flush=True)
    pool.shutdown()
    return out_dir


def cache_text_features(cfg: dict, instructions: list[str], name: str = "lang",
                        task_index: list[int] | None = None) -> Path:
    """CLIP embeddings of `instructions` -> `<cache_dir>/<name>.npz` with keys text, emb (N, 512), task_index."""
    enc = ClipTextEncoder(cfg["encoders"]["text"], "cuda" if torch.cuda.is_available() else "cpu")
    emb = enc(list(instructions)).numpy().astype(np.float32)
    path = cache_dir(cfg) / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    ti = np.arange(len(instructions)) if task_index is None else np.asarray(task_index)
    np.savez(path, text=np.array(instructions), emb=emb, task_index=ti.astype(np.int64))
    return path


def write_meta(cfg: dict, root: Path) -> None:
    stats = json.loads((root / "meta" / "stats.json").read_text())
    meta = {
        "repo_id": cfg["data"]["repo_id"], "snapshot": root.name,
        "vision_encoder": cfg["encoders"]["vision"], "text_encoder": cfg["encoders"]["text"],
        "image_size": cfg["encoders"]["image_size"], "n_aug_views": cfg["data"]["n_aug_views"],
        "augmentation": AUG, "seed": cfg.get("seed", 0), "suite_tasks": {k: list(v) for k, v in SUITE_TASKS.items()},
        "stats": {k: {s: stats[k][s] for s in ("min", "max", "mean", "std")}
                  for k in ("observation.state", "action")},
    }
    path = cache_dir(cfg) / "meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def load_episode(cache: str | Path, ep: int) -> dict[str, np.ndarray]:
    """Read one cached episode (used by ita/data/dataset.py)."""
    with np.load(Path(os.path.expanduser(str(cache))) / "episodes" / f"ep{ep:04d}.npz") as z:
        return {k: z[k] for k in z.files}
