"""Open the LIBERO simulator in a live 3D window (MuJoCo viewer) and move the robot.

The robot follows the same smooth random motion as scripts/02_sim_rollout.py (there is no
trained policy yet); every --steps steps the scene resets to the next fixed initial state.

Run:  python scripts/02b_sim_live.py --suite libero_goal --task-id 0
      python scripts/02b_sim_live.py --camera wrist      # start looking through the wrist camera
Close the window (or press Ctrl+C in the terminal) to stop.

Viewer controls: left-drag rotate · right-drag move · scroll zoom · double-click a body to
focus it · [ and ] switch between the robot's cameras · Esc back to the free camera ·
Tab hides the left panel · Space pauses the physics shown in the window.
Needs a display: on Windows 11, WSLg provides one automatically (DISPLAY=:0).
"""

import argparse
import ctypes
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen camera images; the window itself uses GLFW
# If anything crashes, do not write a core dump: WSL copies it to %LOCALAPPDATA%\Temp\wsl-crashes
# on C: (about 5 GB each). PR_SET_DUMPABLE = 4, value 0 = not dumpable.
ctypes.CDLL("libc.so.6").prctl(4, 0, 0, 0, 0)

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

CONTROL_HZ = 20  # LIBERO's control frequency
CAMERAS = {"scene": "agentview", "wrist": "robot0_eye_in_hand"}


def free_camera_from(cam, model, data, name, distance=1.0):
    """Place the mouse-controlled camera exactly where the fixed camera `name` is, looking the same way.

    MuJoCo's default free camera is framed on the whole world (including the large floor), which in
    LIBERO puts it under the table. A MuJoCo camera looks along -z of its frame; the free camera orbits a
    `lookat` point, so we put that point `distance` metres in front of the fixed camera.
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    forward = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.cam_xpos[cid] + distance * forward
    cam.distance = distance
    cam.azimuth = np.degrees(np.arctan2(forward[1], forward[0]))
    cam.elevation = np.degrees(np.arcsin(forward[2]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_goal")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init-state", type=int, default=0)
    p.add_argument("--steps", type=int, default=300, help="steps before resetting to the next initial state")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0, help="1 = real time, 2 = twice as fast")
    p.add_argument("--camera", choices=["free", *CAMERAS], default="free",
                   help="free = mouse-controlled view; scene / wrist = the robot's cameras")
    args = p.parse_args()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = suite.get_task_init_states(args.task_id)
    print(f"Task {args.task_id} of {args.suite}: '{task.language}'  ({len(init_states)} initial states)")

    # hard_reset=False: env.reset() rewinds the same MuJoCo model instead of rebuilding it, so the
    # viewer stays attached. reset() is needed before every initial state because robosuite stops
    # accepting actions after `horizon` (1000) steps, and set_init_state alone does not reset that count.
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, hard_reset=False)
    env.seed(args.seed)
    env.reset()
    model, data = env.env.sim.model._model, env.env.sim.data._data

    rng = np.random.default_rng(args.seed)
    k = args.init_state
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            first = True
            while viewer.is_running():
                env.reset()
                env.set_init_state(init_states[k % len(init_states)])
                if first:  # set the view once, so a view chosen with the mouse is kept across resets
                    first = False
                    with viewer.lock():
                        if args.camera == "free":
                            free_camera_from(viewer.cam, model, data, CAMERAS["scene"])
                        else:
                            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                                                      CAMERAS[args.camera])
                print(f"initial state {k % len(init_states)}")
                direction = np.zeros(6)
                for t in range(args.steps):
                    if not viewer.is_running():
                        break
                    t0 = time.time()
                    # Same smooth random motion as 02_sim_rollout.py: new direction every 25 steps,
                    # gripper open for the first half, closed for the second.
                    if t % 25 == 0:
                        direction = rng.uniform(-0.5, 0.5, size=6)
                        direction[3:] *= 0.2
                    gripper = -1.0 if t < args.steps // 2 else 1.0
                    _, _, done, _ = env.step(np.append(direction, gripper).tolist())
                    viewer.sync()
                    if done:
                        print(f"  task success at step {t}")
                        break
                    time.sleep(max(0.0, 1 / (CONTROL_HZ * args.speed) - (time.time() - t0)))
                k += 1
    except KeyboardInterrupt:
        pass
    print("closed", flush=True)  # os._exit below skips the normal flush of buffered output
    # Exit without running the EGL/GLFW teardown, which can segfault when both contexts were used.
    # (A segfault inside WSL writes a multi-GB crash dump to %LOCALAPPDATA%\Temp\wsl-crashes.)
    os._exit(0)


if __name__ == "__main__":
    main()
