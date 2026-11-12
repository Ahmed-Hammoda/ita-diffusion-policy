"""Week 1 — are simulator images oriented like the dataset images?

Saves outputs/orientation_check.png with three columns per camera:
    dataset frame | simulator frame (raw) | simulator frame rotated 180°
Whichever simulator column looks like the dataset tells you the value of
`eval.flip_sim_images` in configs/default.yaml (raw → false, rotated → true).

Run:  python scripts/03_check_orientation.py
"""

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from ita.utils import chw_to_hwc_uint8, import_lerobot  # noqa: E402


def main():
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    # dataset frame
    LeRobotDataset, _ = import_lerobot()
    ds = LeRobotDataset("lerobot/libero", episodes=[0])
    sample = ds[0]
    image_keys = sorted(k for k in sample if k.startswith("observation.images"))
    data_imgs = [chw_to_hwc_uint8(sample[k]) for k in image_keys]
    print(f"Dataset episode 0 task: {sample.get('task', '?')}  cameras: {image_keys}")

    # simulator frame (scene differs from the dataset episode, but orientation is obvious: table at the bottom)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    sim_imgs = [obs["agentview_image"], obs["robot0_eye_in_hand_image"]]
    env.close()

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for r, (d_img, s_img, name) in enumerate(zip(data_imgs, sim_imgs, ["scene", "wrist"])):
        for c, (img, title) in enumerate([(d_img, "dataset"), (s_img, "sim raw"),
                                          (np.ascontiguousarray(s_img[::-1, ::-1]), "sim rotated 180°")]):
            axes[r, c].imshow(img)
            axes[r, c].set_title(f"{name}: {title}")
            axes[r, c].axis("off")
    fig.tight_layout()
    path = out / "orientation_check.png"
    fig.savefig(path, dpi=120)
    print(f"Saved {path} — open it and set eval.flip_sim_images in configs/default.yaml")


if __name__ == "__main__":
    main()
