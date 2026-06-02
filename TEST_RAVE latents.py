import torch
import soundfile as sf

from functions.latent_interpolation import load_audio, interpolate_audio

interpolated_path = interpolate_audio(
    "audio_examples/2.wav",
    "audio_examples/1.wav",
    "RAVE Models/voice_vocalset_b2048_r48000_z16.ts",
    amount=0.5,
    output_path="test_interpolation.wav",
)


# Vanilla code
def vanilla_test():
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
    return WAV_PATH


print(f"Interpolated audio saved to: {interpolated_path}")
print(f"Vanilla test audio saved to: {vanilla_test()}")
