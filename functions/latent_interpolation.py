import tempfile
import os

import librosa as li
import soundfile as sf
import numpy as np
import torch

"""functions for encoding, interpolating, and decoding audio in the latent space of a given model"""


def load_audio(audio_path: str) -> torch.tensor:
    """Load audio file and convert to torch tensor."""
    loaded_audio, _ = li.load(audio_path, sr=None)
    return torch.from_numpy(loaded_audio).float().reshape(1, 1, -1)


def array_to_path(
    array: np.array, sample_rate: int = 44100, output_path: str = None
) -> str:
    """
    Takes audio as an array, writes it to a temporary file and passes that file's path back
    """

    if output_path is None:
        # Write output to file
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    else:
        path = output_path

    sf.write(path, array, sample_rate)
    return path


def interpolate_audio(
    audio_path0: str,
    audio_path1: str = None,
    model_string: str = None,
    amount: float = 0,
    output_path: str = None,
) -> str:
    """
    Encodes audio files to latent space of a given model,
    linearly interpolates according to an amount (0 is audio0, 1 is audio1),
    decodes these latents back to audio, writes to temp and returns path.
    """
    gen_model = torch.jit.load(model_string)  # Load Model

    # Load audio and encode
    audio0 = load_audio(audio_path0)

    with torch.no_grad():
        gen_model.eval()
        encoding0 = gen_model.encode(audio0)

    if audio_path1:
        print(
            f"Interpolating between {audio_path0} and {audio_path1} with amount {amount} in model {model_string}"
        )

        audio1 = load_audio(audio_path1)
        with torch.no_grad():
            gen_model.eval()
            encoding1 = gen_model.encode(audio1)
            # print(f"encoding1 shape: {encoding1.shape}")

    else:
        print(
            f"Just one audio provided, generating without interpolation. Size: {audio0.shape}"
        )
        encoding1 = encoding0

    # Interpolate latent2 to match latent1 length
    if encoding0.shape[-1] != encoding1.shape[-1]:
        encoding1 = torch.nn.functional.interpolate(
            encoding1,
            size=encoding0.shape[-1],
            mode="linear",
            align_corners=False,
        )

    interp_encoding = (1 - amount) * encoding0 + amount * encoding1

    interp_encoding = encoding1

    with torch.no_grad():
        output = gen_model.decode(interp_encoding)
    print(output.shape)

    output = torch.mean(output, dim=1, keepdim=True).reshape(-1).numpy()  # mono
    # print(f"Output shape: {output.shape}")

    path = array_to_path(output, output_path=output_path)

    return path
