"""Gradio demo: pick a command, watch the simulated robot execute it.   [Owner: Sim & Demo — week 4]

Layout plan:
  - Dropdown: LIBERO-Goal instructions (+ a free-text box for paraphrases)
  - Dropdown: initial scene index (0-49)
  - Button "Run" -> ita.eval.rollout runs one episode and returns an mp4 path
  - gr.Video output + text: success / failure, number of steps, average inference time

Run:  python demo/app.py   then open http://127.0.0.1:7860
"""

import gradio as gr


def run_episode(instruction: str, init_state: int):
    raise NotImplementedError("TODO: load checkpoint once, call the rollout, return (video_path, summary)")


if __name__ == "__main__":
    with gr.Blocks(title="From Instruction to Action") as app:
        gr.Markdown("# From Instruction to Action\nType or pick a command for the robot.")
        instruction = gr.Textbox(label="Instruction", value="open the middle drawer of the cabinet")
        init_state = gr.Slider(0, 49, step=1, value=0, label="Initial scene")
        run = gr.Button("Run")
        video = gr.Video(label="Rollout")
        summary = gr.Textbox(label="Result")
        run.click(run_episode, [instruction, init_state], [video, summary])
    app.launch()
