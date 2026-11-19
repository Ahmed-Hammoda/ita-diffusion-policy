"""Build the feature cache: frozen DINOv2 features of every frame + CLIP features of the instructions.

Writes to data.cache_dir (configs/default.yaml):
    episodes/epNNNN.npz   scene (T, V, 384) fp16, wrist (T, V, 384) fp16, state (T, 8), action (T, 7)
    lang.npz              the 40 instructions and their CLIP embeddings (40, 512)
    meta.json             encoders, augmentation settings, dataset statistics

Resumable: episodes already in the cache are skipped, so the script can be stopped and restarted.

Run:  python scripts/05_build_cache.py --limit 5             # quick test on the first 5 LIBERO-Goal episodes
      python scripts/05_build_cache.py --suite libero_goal   # one suite (428 episodes)
      python scripts/05_build_cache.py                       # all 4 suites (1693 episodes)
      python scripts/05_build_cache.py --text-only           # only lang.npz
"""

import argparse

from ita.data.embed import (SUITE_TASKS, cache_dir, cache_image_features, cache_text_features, dataset_root,
                            load_lowdim, load_tasks, select_episodes)
from ita.utils import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="config file (default: configs/default.yaml)")
    p.add_argument("--suite", default="all", choices=["all", *SUITE_TASKS])
    p.add_argument("--episodes", type=int, nargs="*", help="explicit episode indices (overrides --suite)")
    p.add_argument("--limit", type=int, default=None, help="only the first N selected episodes (testing)")
    p.add_argument("--overwrite", action="store_true", help="re-encode episodes that are already cached")
    p.add_argument("--text-only", action="store_true", help="only (re)build lang.npz")
    args = p.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    root = dataset_root(cfg["data"]["repo_id"])
    print(f"[cache] dataset: {root}\n[cache] output:  {cache_dir(cfg)}")

    tasks = load_tasks(root)
    path = cache_text_features(cfg, tasks, name="lang")
    print(f"[cache] {len(tasks)} instructions -> {path}")
    if args.text_only:
        return

    if args.episodes:
        episodes = args.episodes
    else:
        suite = "libero_goal" if (args.limit and args.suite == "all") else args.suite
        episodes = select_episodes(load_lowdim(root), suite)
    if args.limit:
        episodes = episodes[:args.limit]
    cache_image_features(cfg, episodes, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
