"""Closed-loop evaluation in the LIBERO simulator.   [Owner: Sim & Eval — weeks 2-3]

For each task and each of its 50 fixed initial states:
  1. env.set_init_state(init_states[i]); run ~10 no-op steps so objects settle
     (no-op action = [0, 0, 0, 0, 0, 0, -1]).
  2. Every `execute_steps` control steps:
       - take the 2 camera images (flip 180° if cfg.eval.flip_sim_images), encode them with the frozen
         DINOv2 exactly like ita/data/embed.py (clean view, no augmentation)
       - build the 8-D state exactly like the dataset (eef pos 3 + axis-angle 3 + gripper qpos 2)
       - normalise with the SAME stats as training
       - policy.predict_action_chunk -> unnormalise -> execute the first `execute_steps` actions
  3. Success = env reports done=True before max_steps.

Metrics: success rate per task and mean; inference time per chunk (ms); save some rollouts as mp4.

Language-grounding experiments (week 3):
  * instruction swap — same initial scene, different command: does the behaviour change?
  * paraphrase — reworded command never seen in training (own paraphrases / LIBERO-plus).
"""

from __future__ import annotations


def evaluate_task(policy, cfg: dict, suite: str, task_id: int, instruction: str | None = None) -> dict:
    """instruction=None uses the task's own command; pass another string for swap / paraphrase tests."""
    raise NotImplementedError("TODO (Sim & Eval)")
