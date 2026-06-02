from functions.text_gen import text_to_audio
import os


def test_text_to_audio(save_file: bool = False) -> None:
    """Test that text_to_audio generates a valid WAV file from text input."""
    test_text = "128 BPM tech house drum loop"
    audio_path = text_to_audio(test_text)

    # Check that the file was created and is a valid WAV file
    assert os.path.exists(audio_path), "Audio file was not created."
    assert audio_path.endswith(".wav"), "Audio file does not have .wav extension."

    if not save_file:
        os.remove(audio_path)
