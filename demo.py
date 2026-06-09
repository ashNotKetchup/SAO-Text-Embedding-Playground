import gradio as gr
import os


from functions.text_gen import text_to_audio
from functions.latent_interpolation import interpolate_audio


def generate_audio(
    text: str,
    model: str,
    length_secs: int = 15,
    interpolation_mode: str = None,
    interpolation_text: str = None,
    interpolation_audio=None,
    interpolation_amount: float = 0.5,
):
    """Generates audio dependent on input"""
    print(
        f"Received inputs - Text: {text}, Model: {model}, Length: {length_secs}s, Interpolation Mode: {interpolation_mode}, Interpolation Text: {interpolation_text}, Interpolation Audio: {interpolation_audio}, Interpolation Amount: {interpolation_amount}"
    )
    text_gen_path = text_to_audio(text, length_secs=length_secs)

    if interpolation_mode == "Text" and interpolation_text:
        interpolation_audio = text_to_audio(interpolation_text, length_secs=length_secs)

    if interpolation_audio:
        interp_path = interpolate_audio(
            text_gen_path, interpolation_audio, model, interpolation_amount
        )
    else:
        interp_path = interpolate_audio(text_gen_path, model_string=model)

    return text_gen_path, interp_path


def update_interpolation_inputs(mode: str):
    show_text = mode == "Text"
    show_audio = mode == "Audio"
    return (
        gr.update(visible=show_text),
        gr.update(visible=show_audio),
    )


# Interface
def build_interface():
    # determine model choices from "RAVE Models" folder (next to this script)
    models_dir = os.path.join(os.path.dirname(""), "RAVE Models")
    model_choices = []
    try:
        if os.path.isdir(models_dir):
            model_choices = sorted(
                [
                    os.path.join(models_dir, f)
                    for f in os.listdir(models_dir)
                    if not f.startswith(".")
                ]
            )
    except Exception:
        model_choices = []

    if not model_choices:
        model_choices = ["No Models Found"]

    with gr.Blocks() as demo:
        gr.Markdown(
            "# Text-Timbre Playground "
            "\nGenerate audio from text and interpolate in the latent space of RAVE models (In `RAVE Models` folder). "
            "\n\n- Enter a text prompt and select a length to generate audio from the Stable Audio Open 1.0 model. "
            "\n- Optionally, provide a second text prompt or an audio file to interpolate with, and select a RAVE model for interpolation. "
            "\n- Adjust the interpolation amount to blend between the original and target audio. "
        )
        with gr.Row():
            with gr.Column():
                txt = gr.Textbox(
                    label="Create a sample that sounds like...",
                    placeholder="Enter prompt for audio/music generation",
                )
                length_secs = gr.Slider(
                    minimum=1,
                    maximum=60,
                    value=15,
                    step=1,
                    label="Length (seconds)",
                )
                text_gen_out = gr.Audio(label="Text generation output")
                model = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0],
                    label="Interpolation model",
                )
                interp_out = gr.Audio(label="Interpolation output")
            with gr.Column():
                with gr.Group():
                    interp_mode = gr.Dropdown(
                        choices=["Text", "Audio"],
                        value="Text",
                        label="Interpolation source",
                    )
                    interp_txt = gr.Textbox(
                        label="Interpolation text (optional)",
                        placeholder="Optional text to interpolate from",
                        visible=True,
                    )
                    interp_audio = gr.Audio(
                        label="Interpolation audio (optional)",
                        sources=["upload"],
                        type="filepath",
                        visible=False,
                    )
                    interp_amount = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Interpolation amount",
                    )
                btn = gr.Button("Generate")

        interp_mode.change(
            fn=update_interpolation_inputs,
            inputs=interp_mode,
            outputs=[interp_txt, interp_audio],
        )
        btn.click(
            fn=generate_audio,
            inputs=[
                txt,
                model,
                length_secs,
                interp_mode,
                interp_txt,
                interp_audio,
                interp_amount,
            ],
            outputs=[text_gen_out, interp_out],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(debug=True)
