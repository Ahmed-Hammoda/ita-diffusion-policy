# From Instruction to Action

**Language-conditioned robot manipulation with a Transformer diffusion policy trained from scratch.**
CS495 Deep Learning project, Mediterranean Institute of Technology (MedTech).

A Franka Panda arm in the LIBERO simulator receives a typed command (e.g. *"open the middle drawer of the cabinet"*),
looks at the scene through two cameras, and outputs a chunk of 16 future actions by iteratively denoising random noise.

---

## Architecture

```
 instruction text      scene cam        wrist cam       robot state (8-D)
       │                   │                │                  │
 [CLIP text enc.]    [DINOv2 ViT-S]   [DINOv2 ViT-S]           │
   frozen, 512-d       frozen, 384-d    frozen, 384-d          │
       └──────────── linear projections → d_model ─────────────┘
                                │
                ┌───────────────▼───────────────┐
                │ Transformer ENCODER (ours)    │  self-attention over observation tokens
                └───────────────┬───────────────┘
                                │ memory
 noisy actions (16×7) ──► ┌─────▼──────────────────────────┐
 + timestep k             │ Transformer DECODER (ours)     │  self-attn + cross-attn
                          └─────┬──────────────────────────┘
                                ▼
                    predicted noise ε → DDIM (10 steps) → action chunk
```

| Component | Pretrained? | Trained by us? | Needs a dataset? |
|---|---|---|---|
| DINOv2 ViT-S/14 (`facebook/dinov2-small`) – image features | yes | no (frozen) | no – only run once over LIBERO frames to cache features |
| CLIP ViT-B/32 text encoder (`openai/clip-vit-base-patch32`) – instruction features | yes | no (frozen) | no – only 40 distinct instructions, encoded once |
| **Encoder–decoder Transformer + diffusion head** | **no** | **yes, from scratch** | **yes – LIBERO (`lerobot/libero`)** |

Written by hand (no `nn.Transformer`, no `nn.MultiheadAttention`): attention, positional encoding, feed-forward,
encoder/decoder layers, DDPM/DDIM noise scheduler.

---

## Project structure

```
configs/default.yaml        all hyper-parameters in one place
ita/                        our Python package ("instruction-to-action")
  data/                     dataset → (obs history, 16-step action chunk) pairs, normalisation, embedding cache
  models/                   transformer layers (from scratch) + DiffusionPolicy
  diffusion/                DDPM training noise / DDIM sampling
  eval/                     simulator rollouts, success rate, language-grounding tests
  utils.py                  config, seeding, LeRobot import helper
scripts/
  00_check_env.py           week 1 – is everything installed? GPU visible? MuJoCo renders?
  01_explore_dataset.py     week 1 – download a few LIBERO episodes, print keys/shapes/stats, save frames
  02_sim_rollout.py         week 1 – run the LIBERO simulator, save a video, print observation keys
  03_check_orientation.py   week 1 – compare dataset image vs simulator image (upside-down check)
demo/app.py                 Gradio demo (week 7)
tests/                      unit tests = specification for the from-scratch modules
notebooks/                  exploration notebooks
outputs/                    videos, checkpoints, plots (git-ignored)
```

---

## Setup (do this on every team member's machine)

LIBERO (robosuite + MuJoCo) and LeRobot target Linux, and Kaggle/Colab are Linux too,
so on Windows we work inside **WSL2 + Ubuntu 22.04**. The NVIDIA GPU is usable from WSL2.

### 1. Windows side
```powershell
wsl --install -d Ubuntu-22.04      # admin PowerShell, then reboot
```
- Install the VS Code extension **WSL** (ms-vscode-remote.remote-wsl).
- Create accounts: Hugging Face, Kaggle, Google Colab, (optional) Weights & Biases.

### 2. Ubuntu (WSL) side
```bash
sudo apt update && sudo apt install -y build-essential git cmake ffmpeg libegl1 libgl1 libosmesa6-dev

# Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh      # accept, then restart the terminal

conda create -n ita python=3.10 -y
conda activate ita

# Keep the code in the Linux filesystem (~/), NOT /mnt/c/... — disk access from /mnt/c is very slow.
cd ~
git clone <our-github-repo-url> Instruction-to-Action
git clone https://github.com/huggingface/lerobot.git
pip install -e "lerobot[libero]"            # dataset tools + LIBERO simulator (check docs if the extra name changed)

cd Instruction-to-Action
pip install -r requirements.txt
pip install -e .                            # makes `import ita` work everywhere

echo 'export MUJOCO_GL=egl' >> ~/.bashrc    # headless rendering; use osmesa if egl fails (slower, CPU)
source ~/.bashrc
```
Then open the folder from VS Code with **WSL: Open Folder in WSL**.

> If LIBERO asks on first import *"Do you want to specify a custom path for the dataset folder? (Y/N)"*, answer **N**.

### 3. Verify
```bash
python scripts/00_check_env.py          # every line must say [ OK ] and CUDA=True
```

---

## Week-1 checklist

- [ ] `python scripts/00_check_env.py` – all OK on every machine
- [ ] `python scripts/01_explore_dataset.py` – dataset keys, shapes, tasks and stats printed; frames saved to `outputs/explore/`
- [ ] `python scripts/02_sim_rollout.py --suite libero_goal --task-id 0` – video in `outputs/sim/`
- [ ] `python scripts/03_check_orientation.py` – decide whether simulator images must be flipped 180° to match the dataset
- [ ] Write down how the 8-D dataset state maps to simulator observations (eef pos 3 + axis-angle 3 + gripper 2)

## Known pitfalls

1. **Image orientation** – raw LIBERO renders are upside-down; the dataset may already be flipped. Check with script 03.
2. **State/action format** – at evaluation time, build the 8-D state from the simulator *exactly* like the dataset did, and use the *same* normalisation stats.
3. **Augmentation vs cached embeddings** – augmentations must be applied *before* DINOv2, so cache several augmented views per frame (`data.n_aug_views`).
4. **Paraphrases** – with only 10–40 distinct instructions, the model can memorise embeddings; keep test paraphrases separate from any used in training.

---

## Team roles

| Role | Owns |
|---|---|
| Data | LeRobot loading, normalisation, chunk pairs, episode split, embedding cache (`ita/data/`) |
| Model | From-scratch Transformer + unit tests (`ita/models/`, `tests/`) |
| Diffusion & Training | DDPM/DDIM, training loop, Kaggle/Colab runs, regression baseline (`ita/diffusion/`) |
| Sim, Eval & Demo | LIBERO rollouts, success rate, instruction-swap & paraphrase tests, Gradio (`ita/eval/`, `demo/`) |

## Timeline

| Week | Milestone |
|---|---|
| 1 | Environments working, simulator video, dataset downloaded, orientation checked |
| 2 | Data pipeline + embedding cache; transformer layers written and unit-tested |
| 3 | Full model overfits a tiny subset; diffusion sampling works |
| 4 | Train on 3–5 LIBERO-Goal tasks → first success rates |
| 5 | All 10 Goal tasks, baseline comparison, ablations (DDIM steps, history length) |
| 6 | Language grounding: instruction swap + paraphrases |
| 7 | Scale to all 40 tasks; Gradio demo |
| 8 | Report, plots, demo video |

## References

LIBERO (Liu et al., 2023) · Diffusion Policy (Chi et al., 2023) · ACT (Zhao et al., 2023) · SmolVLA (Shukor et al., 2025) ·
Attention Is All You Need (Vaswani et al., 2017) · CLIP (Radford et al., 2021) · DINOv2 (Oquab et al., 2024) ·
LeRobot LIBERO docs: https://huggingface.co/docs/lerobot/libero
