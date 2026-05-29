import torch
import numpy
import librosa as li
import soundfile as sf


def load_audio(audio_path: str) -> torch.tensor:
    """Load audio file and convert to torch tensor."""
    loaded_audio, _ = li.load(audio_path, sr=None)
    return torch.from_numpy(loaded_audio).float().reshape(1, 1, -1)


MODEL_STRING = "RAVE Models/voice_vocalset_b2048_r48000_z16.ts"
AUDIO_PATH = "audio_examples/2.wav"
WAV_PATH = "test.wav"

gen_model = torch.jit.load(MODEL_STRING)  # Load Model
audio = load_audio(AUDIO_PATH)

with torch.no_grad():
    gen_model.eval()
    latent = gen_model.encode(audio)

with torch.no_grad():
    output = gen_model.decode(latent)

output = torch.mean(output, dim=1, keepdim=True).reshape(-1).numpy()  # mono

sf.write(WAV_PATH, output, 44100)

print(f"Output shape: {output.shape}")
