"""Week 1 — check that everything is installed, the GPU is visible and MuJoCo can render headless.

Run:  python scripts/00_check_env.py
"""

import importlib
import os
import platform
import sys

os.environ.setdefault("MUJOCO_GL", "egl")


def check(name, fn) -> bool:
    try:
        print(f"[ OK ] {name}: {fn()}")
        return True
    except Exception as e:  # noqa: BLE001 — we want to report every failure
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def version(module):
    def f():
        m = importlib.import_module(module)
        return getattr(m, "__version__", "imported")
    return f


def torch_info():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} installed but CUDA is NOT available")
    props = torch.cuda.get_device_properties(0)
    return f"torch {torch.__version__}, GPU = {props.name}, {props.total_memory / 2**30:.1f} GB"


def mujoco_render():
    import mujoco
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><light pos='0 0 1'/><geom size='.1'/></worldbody></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 64, 64)
    renderer.update_scene(data)
    img = renderer.render()
    return f"mujoco {mujoco.__version__}, rendered {img.shape} with MUJOCO_GL={os.environ['MUJOCO_GL']}"


def libero():
    from libero.libero import benchmark
    suites = list(benchmark.get_benchmark_dict().keys())
    return f"benchmarks: {suites}"


def lerobot_dataset_class():
    from ita.utils import import_lerobot
    import_lerobot()
    return "LeRobotDataset importable"


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} on {platform.system()} {platform.release()}\n")
    results = [
        check("PyTorch + CUDA", torch_info),
        check("transformers", version("transformers")),
        check("lerobot", version("lerobot")),
        check("LeRobotDataset", lerobot_dataset_class),
        check("robosuite", version("robosuite")),
        check("MuJoCo headless render", mujoco_render),
        check("LIBERO", libero),
        check("gradio", version("gradio")),
        check("imageio", version("imageio")),
    ]
    print(f"\n{sum(results)}/{len(results)} checks passed")
    if platform.system() == "Windows":
        print("Note: you are on native Windows — the simulator is expected to run inside WSL2 (see README).")
    sys.exit(0 if all(results) else 1)
