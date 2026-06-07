import tempfile
import os
import soundfile as sf
import torch
import torchaudio

# From Stable audio open page on huggingface (https://huggingface.co/stabilityai/stable-audio-open-1.0):

import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


def text_to_audio(text: str, length_secs: int = 15) -> str:
    """Audio generator: create a WAV file from text -> audio model and return its path."""

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    model = model.to(device)

    # Set up text and timing conditioning
    conditioning = [
        {
            "prompt": text,
            "seconds_start": 0,
            "seconds_total": length_secs,
        }
    ]

    # Generate stereo audio
    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device,
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, convert to int16, and save to file
    output = (
        output.to(torch.float32)
        .div(torch.max(torch.abs(output)))
        .clamp(-1, 1)
        .mul(32767)
        .to(torch.int16)
        .cpu()[:, : (length_secs * sample_rate)]  # Trim to length_secs
    )

    torchaudio.save(path, output, sample_rate)

    return path
