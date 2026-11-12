"""Week 1 — run the LIBERO simulator with simple scripted/random actions and save a video.

Also prints every observation key (needed to rebuild the 8-D state like the dataset).

Run:  python scripts/02_sim_rollout.py --suite libero_goal --task-id 0
Suites: libero_goal, libero_spatial, libero_object, libero_10, libero_90
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-state", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/sim")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    print(f"Suite {args.suite}: {suite.n_tasks} tasks")
    for i in range(suite.n_tasks):
        print(f"  [{i}] {suite.get_task(i).language}")

    task = suite.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"\nRunning task {args.task_id}: '{task.language}'")

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    env.reset()
    init_states = suite.get_task_init_states(args.task_id)
    print(f"{len(init_states)} fixed initial states available for this task")
    obs = env.set_init_state(init_states[args.init_state])

    print("\nObservation keys:")
    for k, v in obs.items():
        print(f"  {k:35s} {np.asarray(v).shape}")

    # Smooth random motion: pick a new random direction every 25 steps, toggle the gripper halfway.
    # Action = [dx, dy, dz, droll, dpitch, dyaw, gripper]  (gripper -1 = open, +1 = close)
    rng = np.random.default_rng(args.seed)
    frames, direction = [], np.zeros(6)
    for t in range(args.steps):
        if t % 25 == 0:
            direction = rng.uniform(-0.5, 0.5, size=6)
            direction[3:] *= 0.2
        gripper = -1.0 if t < args.steps // 2 else 1.0
        obs, reward, done, info = env.step(np.append(direction, gripper).tolist())
        frames.append(np.concatenate([obs["agentview_image"], obs["robot0_eye_in_hand_image"]], axis=1))
        if done:
            print(f"(task reported success at step {t} — lucky!)")
            break

    video = out / f"{args.suite}_task{args.task_id}_random.mp4"
    imageio.mimsave(video, frames, fps=20)
    imageio.imwrite(out / f"{args.suite}_task{args.task_id}_first_frame.png", frames[0])
    env.close()

    print(f"\nSaved {video}  (left: scene camera, right: wrist camera)")
    print("If the image looks upside-down, that is the known LIBERO rendering quirk — "
          "run scripts/03_check_orientation.py.")
    print("\nState hint: dataset state (8) is usually eef_pos(3) + axis-angle from eef_quat(3) + gripper_qpos(2).")
    print(f"  robot0_eef_pos      = {np.round(obs['robot0_eef_pos'], 3)}")
    print(f"  robot0_eef_quat     = {np.round(obs['robot0_eef_quat'], 3)}")
    print(f"  robot0_gripper_qpos = {np.round(obs['robot0_gripper_qpos'], 3)}")


if __name__ == "__main__":
    main()
