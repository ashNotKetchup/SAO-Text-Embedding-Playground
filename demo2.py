import tempfile
import gradio as gr
import os

import librosa as li
import soundfile as sf
import numpy as np
import torch
import torchaudio
import wave
import struct


def load_audio(audio_path: str) -> torch.tensor:
    """Load audio file and convert to torch tensor."""
    audio, _ = li.load(audio_path, sr=None)
    return torch.from_numpy(audio).float()


def text_to_audio(text: str) -> str:
    """Proxy audio generator: create a short silent WAV file and return its path.

    Placeholder for future music generation.
    """
    if not text:
        return None
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    # Placeholder for generation - load example audio
    audio, sample_rate = li.load("audio_examples/1.wav", sr=None)
    sf.write(path, audio, sample_rate)

    return path


def array_to_path(array: np.array, sample_rate: int = 44100) -> str:
    """
    Takes audio as an array, writes it to a temporary file and passes that file's path back
    """

    # Write output to file
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        silent_frame = struct.pack("<h", 0)
        wf.writeframes(array)

    return path


def interpolate_audio(
    audio_path0: str,
    audio_path1: str = None,
    model_string: str = None,
    amount: float = 0,
) -> str:
    """
    Encodes audio files to latent space of a given model,
    linearly interpolates according to an amount (0 is audio0, 1 is audio1),
    decodes these latents back to audio
    """
    gen_model = torch.jit.load(model_string)  # Load Model
    print(
        f"Interpolating between {audio_path0} and {audio_path1} with amount {amount} in model {model_string}"
    )

    audio1 = load_audio(audio_path0)
    audio1 = audio1.reshape(1, 1, -1)

    with torch.no_grad():
        gen_model.eval()
        encoding1 = gen_model.encode(audio1)

    if audio_path1:
        audio2 = load_audio(audio_path1).reshape(1, 1, -1)
        with torch.no_grad():
            gen_model.eval()
            encoding2 = gen_model.encode(audio2)

    else:
        print(
            f"Just one audio provided, generating without interpolation. Size: {audio1.shape}"
        )
        encoding2 = encoding1

    # Interpolate latent2 to match latent1 length
    if encoding1.shape[-1] != encoding2.shape[-1]:
        encoding2 = torch.nn.functional.interpolate(
            encoding2,
            size=encoding1.shape[-1],
            mode="linear",
            align_corners=False,
        ).squeeze(0)

    interp_encoding = (1 - amount) * encoding1 + amount * encoding2

    with torch.no_grad():
        output = gen_model.decode(interp_encoding)
    print(output.shape())
    output = torch.mean(output, dim=1, keepdim=True).reshape(-1).numpy()  # mono
    print(f"Output shape: {output.shape}")

    path = array_to_path(output)

    return path


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

    else:
        return interpolate_audio(text_to_audio(text), model_string=model)

    return


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
