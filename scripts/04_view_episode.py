"""Export one LIBERO demonstration so it can be opened from Windows.

Writes to outputs/episodes/:
    ep<N>.mp4   scene camera | wrist camera, side by side, 10 fps
    ep<N>.csv   one row per frame: 8-D state and 7-D action
    ep<N>.txt   the instruction and basic information

Run:  python scripts/04_view_episode.py --episode 0
      python scripts/04_view_episode.py --task-index 19      # first demo of dataset task 19
      python scripts/04_view_episode.py --list-tasks         # task numbers, instructions, demo counts
"""

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from ita.utils import chw_to_hwc_uint8, import_lerobot

REPO = "lerobot/libero"
STATE_NAMES = ["eef_x", "eef_y", "eef_z", "rot_ax", "rot_ay", "rot_az", "finger_1", "finger_2"]
ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def episode_table():
    """(episode_index, task_index) for every episode, plus the task strings."""
    LeRobotDataset, LeRobotDatasetMetadata = import_lerobot()
    meta = LeRobotDatasetMetadata(REPO)
    ds = LeRobotDataset(REPO)
    df = ds.hf_dataset.select_columns(["episode_index", "task_index"]).to_pandas()
    per_episode = df.drop_duplicates("episode_index")
    tasks = meta.tasks
    if hasattr(tasks, "index") and hasattr(tasks, "columns"):  # DataFrame indexed by task string
        names = {int(row.task_index): str(t) for t, row in tasks.iterrows()}
    else:
        names = {int(k): str(v) for k, v in dict(tasks).items()}
    return per_episode, names


def list_tasks():
    per_episode, names = episode_table()
    counts = per_episode.groupby("task_index")["episode_index"].agg(["count", "min"])
    suites = {0: "LIBERO-10", 1: "LIBERO-Goal", 2: "LIBERO-Object", 3: "LIBERO-Spatial"}
    print(f"{'task':>4}  {'suite':<14} {'demos':>5} {'first ep':>8}  instruction")
    for t, row in counts.iterrows():
        print(f"{t:>4}  {suites.get(t // 10, '?'):<14} {row['count']:>5} {row['min']:>8}  {names.get(t, '?')}")


def export(episode: int, out_dir: Path):
    LeRobotDataset, _ = import_lerobot()
    ds = LeRobotDataset(REPO, episodes=[episode])
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, rows, task = [], [], ""
    for i in range(len(ds)):
        s = ds[i]
        task = s.get("task", task)
        frames.append(np.concatenate([chw_to_hwc_uint8(s["observation.images.image"]),
                                      chw_to_hwc_uint8(s["observation.images.image2"])], axis=1))
        rows.append([i] + s["observation.state"].tolist() + s["action"].tolist())

    stem = out_dir / f"ep{episode:04d}"
    imageio.mimsave(stem.with_suffix(".mp4"), frames, fps=10)
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame"] + [f"state_{n}" for n in STATE_NAMES] + [f"action_{n}" for n in ACTION_NAMES])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.5f}" for v in r[1:]])
    stem.with_suffix(".txt").write_text(
        f"episode: {episode}\ninstruction: {task}\nframes: {len(frames)} ({len(frames) / 10:.1f} s at 10 Hz)\n"
        "video: left = scene camera, right = wrist camera\n", encoding="utf-8")

    print(f"Episode {episode}: '{task}', {len(frames)} frames")
    for ext in (".mp4", ".csv", ".txt"):
        print(f"  saved {stem.with_suffix(ext)}")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--episode", type=int, default=0)
    g.add_argument("--task-index", type=int, help="export the first demonstration of this dataset task")
    g.add_argument("--list-tasks", action="store_true")
    p.add_argument("--out", default="outputs/episodes")
    args = p.parse_args()

    if args.list_tasks:
        list_tasks()
        return
    episode = args.episode
    if args.task_index is not None:
        per_episode, _ = episode_table()
        matches = per_episode[per_episode.task_index == args.task_index].episode_index
        if matches.empty:
            raise SystemExit(f"No episodes for task {args.task_index}")
        episode = int(matches.min())
    export(episode, Path(args.out))


if __name__ == "__main__":
    main()
