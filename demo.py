import gradio as gr
import os


from functions.text_gen import text_to_audio
from functions.latent_interpolation import interpolate_audio


def generate_audio(
    text: str,
    model: str,
    interpolation_mode: str = None,
    interpolation_text: str = None,
    interpolation_audio=None,
    interpolation_amount: float = 0.5,
):
    """Generates audio dependent on input"""
    print(
        f"Received inputs - Text: {text}, Model: {model}, Interpolation Mode: {interpolation_mode}, Interpolation Text: {interpolation_text}, Interpolation Audio: {interpolation_audio}, Interpolation Amount: {interpolation_amount}"
    )
    if interpolation_mode == "Text" and interpolation_text:
        print("Performing text-based interpolation (placeholder)")
        interpolation_audio = text_to_audio(interpolation_text)

    if interpolation_audio:
        return interpolate_audio(
            text_to_audio(text), interpolation_audio, model, interpolation_amount
        )

    return interpolate_audio(text_to_audio(text), model_string=model)


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
    models_dir = os.path.join(os.path.dirname(__file__), "RAVE Models")
    model_choices = []
    try:
        if os.path.isdir(models_dir):
            # list non-hidden files/directories and create full paths
            model_choices = sorted(
                [
                    os.path.join(models_dir, f)
                    for f in os.listdir(models_dir)
                    if not f.startswith(".")
                ]
            )
    except Exception:
        model_choices = []

    # fallback to defaults if folder missing or empty
    if not model_choices:
        model_choices = ["gpt-4o-mini-tts", "gpt-4o-mini", "gpt-4o"]

    with gr.Blocks() as demo:
        gr.Markdown("# Audio Generation Demo")
        with gr.Row():
            with gr.Column():
                txt = gr.Textbox(
                    label="Input text",
                    placeholder="Enter prompt for audio/music generation",
                )
                model = gr.Dropdown(
                    choices=model_choices, value=model_choices[0], label="Model"
                )
                audio = gr.Audio(label="Audio output")
                btn = gr.Button("Generate")
            with gr.Column():
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

        interp_mode.change(
            fn=update_interpolation_inputs,
            inputs=interp_mode,
            outputs=[interp_txt, interp_audio],
        )
        btn.click(
            fn=generate_audio,
            inputs=[txt, model, interp_mode, interp_txt, interp_audio, interp_amount],
            outputs=audio,
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
