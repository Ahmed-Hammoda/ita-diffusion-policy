"""Week 1 — download a few LIBERO episodes (LeRobot format) and look at what is inside.

Prints: number of episodes/frames, fps, feature keys and shapes, the task instructions,
action/state statistics; saves both camera images of a few frames to outputs/explore/.

Run:  python scripts/01_explore_dataset.py                 # episode 0 only (small download)
      python scripts/01_explore_dataset.py --episodes 0 1 2
      python scripts/01_explore_dataset.py --all             # whole dataset (~1.9 GB)
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from ita.utils import chw_to_hwc_uint8, import_lerobot, task_list


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="lerobot/libero")
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--all", action="store_true", help="download every episode")
    p.add_argument("--out", default="outputs/explore")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    LeRobotDataset, LeRobotDatasetMetadata = import_lerobot()

    # ---- metadata (tiny download) ----
    meta = LeRobotDatasetMetadata(args.repo_id)
    print("=" * 70)
    print(f"Dataset   : {args.repo_id}")
    print(f"Episodes  : {meta.total_episodes}")
    print(f"Frames    : {meta.total_frames}")
    print(f"FPS       : {meta.fps}")
    print("\nFeatures:")
    for key, ft in meta.features.items():
        print(f"  {key:35s} dtype={ft.get('dtype')!s:8s} shape={ft.get('shape')}")

    tasks = task_list(meta)
    print(f"\n{len(tasks)} task instructions:")
    for i, t in enumerate(tasks):
        print(f"  [{i:2d}] {t}")
    (out / "tasks.txt").write_text("\n".join(tasks), encoding="utf-8")

    for key in ("observation.state", "action"):
        if key in meta.stats:
            s = meta.stats[key]
            print(f"\nStats of {key}:")
            for stat in ("min", "max", "mean", "std"):
                if stat in s:
                    print(f"  {stat:4s} {np.round(np.asarray(s[stat]), 3)}")

    # ---- actual frames ----
    episodes = None if args.all else args.episodes
    ds = LeRobotDataset(args.repo_id, episodes=episodes)
    print("\n" + "=" * 70)
    print(f"Loaded {ds.num_episodes} episode(s), {ds.num_frames} frames")

    sample = ds[0]
    print("\nOne sample (frame 0):")
    for k, v in sample.items():
        shape = tuple(v.shape) if hasattr(v, "shape") else ""
        print(f"  {k:35s} {type(v).__name__:8s} {shape} {'' if shape else repr(v)[:80]}")

    image_keys = [k for k in sample if k.startswith("observation.images")]
    n = len(ds)
    for idx in sorted({0, n // 2, n - 1}):
        s = ds[idx]
        row = np.concatenate([chw_to_hwc_uint8(s[k]) for k in image_keys], axis=1)
        path = out / f"frame_{idx:06d}.png"
        imageio.imwrite(path, row)
        print(f"Saved {path}  ({' | '.join(image_keys)})  task: {s.get('task', '?')}")


if __name__ == "__main__":
    main()
