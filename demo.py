
import tempfile
import gradio as gr
import os
import wave
import struct



def text_to_speech(text: str, model: str = "nasa"):
	"""Proxy audio generator: create a short silent WAV file and return its path.

	Placeholder for future music generation. The `model` parameter is
	accepted from the UI but not used by this proxy implementation.
	"""
	if not text:
		return None
	fd, path = tempfile.mkstemp(suffix=".wav")
	os.close(fd)

	# create 1 second of silence (16-bit PCM, 16kHz, mono)
	sample_rate = 16000
	duration_s = 1
	n_frames = sample_rate * duration_s
	with wave.open(path, "w") as wf:
		wf.setnchannels(1)
		wf.setsampwidth(2)
		wf.setframerate(sample_rate)
		silent_frame = struct.pack('<h', 0)
		wf.writeframes(silent_frame * n_frames)

	return path


def generate_audio(text: str, model: str, interpolation_mode: str = None, interpolation_text: str = None, interpolation_audio=None, interpolation_amount: float = 0.5):
	"""Proxy wrapper for future interpolation support.

	Interpolation inputs are accepted by the UI and reserved for later use.
	"""
	return text_to_speech(text=text, model=model)


def update_interpolation_inputs(mode: str):
	show_text = mode == "Text"
	show_audio = mode == "Audio"
	return (
		gr.update(visible=show_text),
		gr.update(visible=show_audio),
	)


def build_interface():
	# determine model choices from "RAVE Models" folder (next to this script)
	models_dir = os.path.join(os.path.dirname(__file__), "RAVE Models")
	model_choices = []
	try:
		if os.path.isdir(models_dir):
			# list non-hidden files/directories
			model_choices = sorted([f for f in os.listdir(models_dir) if not f.startswith('.')])
	except Exception:
		model_choices = []

	# fallback to defaults if folder missing or empty
	if not model_choices:
		model_choices = ["gpt-4o-mini-tts", "gpt-4o-mini", "gpt-4o"]

	with gr.Blocks() as demo:
		gr.Markdown("# Audio Generation Demo")
		with gr.Row():
			with gr.Column():
				txt = gr.Textbox(label="Input text", placeholder="Enter prompt for audio/music generation")
				model = gr.Dropdown(choices=model_choices, value=model_choices[0], label="Model")
				audio = gr.Audio(label="Audio output")
				btn = gr.Button("Generate")
			with gr.Column():
				interp_mode = gr.Dropdown(choices=["Text", "Audio"], value="Text", label="Interpolation source")
				interp_txt = gr.Textbox(label="Interpolation text (optional)", placeholder="Optional text to interpolate from", visible=True)
				interp_audio = gr.Audio(label="Interpolation audio (optional)", sources=["upload"], type="filepath", visible=False)
				interp_amount = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="Interpolation amount")

		interp_mode.change(fn=update_interpolation_inputs, inputs=interp_mode, outputs=[interp_txt, interp_audio])
		btn.click(fn=generate_audio, inputs=[txt, model, interp_mode, interp_txt, interp_audio, interp_amount], outputs=audio)

	return demo


if __name__ == "__main__":
	demo = build_interface()
	demo.launch()
