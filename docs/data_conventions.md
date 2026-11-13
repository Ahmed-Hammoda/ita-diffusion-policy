# Data and simulator conventions

Findings from Step 3 (`scripts/01`–`03`), checked against LeRobot's LIBERO integration
(`lerobot/sim/integrations/libero.py`, LeRobot 0.7.1). The training data pipeline
(`ita/data/`) and the simulator evaluation (`ita/eval/rollout.py`) must both follow them.

## Dataset: `lerobot/libero`

| Key | Shape | Meaning |
|---|---|---|
| `observation.images.image` | 3 × 256 × 256 | Scene camera (simulator `agentview_image`) |
| `observation.images.image2` | 3 × 256 × 256 | Wrist camera (simulator `robot0_eye_in_hand_image`) |
| `observation.state` | 8 | Robot state, see below |
| `action` | 7 | End-effector delta command, see below |
| `task` | string | Instruction of the episode |

1,693 episodes, 273,465 frames at 10 Hz. Images are stored as video and decoded by LeRobot
as float tensors in [0, 1]. Dataset images are upright.

### Task indices and suites

`task_index` groups the 40 instructions by suite in blocks of ten:

| `task_index` | Suite |
|---|---|
| 0–9 | LIBERO-10 (Long) |
| 10–19 | LIBERO-Goal |
| 20–29 | LIBERO-Object |
| 30–39 | LIBERO-Spatial |

The order inside a block is **not** the simulator's task order (dataset index 19 is
"open the middle drawer of the cabinet", which is LIBERO-Goal task 0 in the simulator).
Match dataset episodes to simulator tasks by the instruction string, never by index.

## State (8-D)

```
state = [ eef_pos (3) | axis_angle(eef_quat) (3) | gripper_qpos (2) ]
```

| Dims | Simulator source | Dataset range |
|---|---|---|
| 0–2 | `robot0_eef_pos` (metres) | x −0.48…0.21, y −0.33…0.39, z 0.01…1.37 |
| 3–5 | `robot0_eef_quat` (x, y, z, w) converted to axis-angle | about −3.6…3.7 |
| 6–7 | `robot0_gripper_qpos` (two finger joints) | 0…0.042 and −0.042…0 |

Quaternion to axis-angle, as in robosuite's `quat2axisangle`:

```python
w = clip(q[3], -1, 1)
den = sqrt(max(1 - w*w, 0))
axis_angle = q[:3] * (2 * arccos(w) / den) if den > 1e-10 else zeros(3)
```

## Action (7-D)

```
action = [ dx, dy, dz (3) | droll, dpitch, dyaw (3) | gripper (1) ]
```

- Sent to `env.step()` unchanged; no rescaling between dataset and simulator.
- Dataset range: position about ±0.94, rotation about ±0.38, gripper exactly −1 or +1.
- Gripper: −1 = open, +1 = close.
- No-op action (used to let objects settle after a reset): `[0, 0, 0, 0, 0, 0, -1]`.

## Simulator images

Raw LIBERO renders are rotated 180° relative to the dataset (robot at the bottom of the
scene view, gripper at the top of the wrist view). Rotate every simulator frame before
encoding it:

```python
img = img[::-1, ::-1]
```

This is `eval.flip_sim_images: true` in `configs/default.yaml`.

## Evaluation protocol details

- 50 fixed initial states per task, from `suite.get_task_init_states(task_id)`.
- After `env.set_init_state(...)`, step the no-op action 10 times before the policy acts.
- Success is `done == True` from `env.step()`.
