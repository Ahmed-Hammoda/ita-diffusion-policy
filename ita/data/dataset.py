"""Training pairs: (observation history, 16-step future action chunk).   [Owner: Data — week 2]

One sample at frame t of an episode:

    obs = {
        "img_scene": (T_o, 384)   cached DINOv2 features of the scene camera, frames t-T_o+1 .. t
        "img_wrist": (T_o, 384)   cached DINOv2 features of the wrist camera
        "state":     (T_o, 8)     normalised robot state
        "lang":      (512,)       cached CLIP text feature of the episode instruction
    }
    actions = (H, 7)              normalised actions t .. t+H-1   (H = action_horizon = 16)

Rules:
  * Never cross episode boundaries. At the start of an episode repeat the first frame (pad left);
    near the end repeat the last action (pad right) and return an `action_mask` (H,) of valid steps.
  * Train/validation split is done by EPISODE, not by frame.
  * During training pick one of the `n_aug_views` cached augmented image features at random.
"""

from __future__ import annotations

from torch.utils.data import Dataset


class ChunkDataset(Dataset):
    def __init__(self, cache_dir: str, episode_ids: list[int], obs_history: int = 2,
                 action_horizon: int = 16, train: bool = True):
        raise NotImplementedError("TODO (Data): load cached features + index (episode, frame) pairs")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError
