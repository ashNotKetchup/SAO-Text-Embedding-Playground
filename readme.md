# Text-Timbre Playground


<a target="_blank" href="https://colab.research.google.com/github/ashNotKetchup/SAO-Text-Embedding-Playground/blob/main/demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This repo provides a few methods for playing with systems for generating audio samples.

With these, you can perform:
- Generation of audio from text prompts (text → audio; using Stable Audio Open 1.0)
- Timbre transfer of generated or submitted audio samples (audio → model → audio; uses any model selected from the `RAVE Models` folder, presuming it has a `encode` and `decode` function)
- Interpolation between audio samples (audio1 + audio2 → model → audio; uses linear interpolation in the latent/embedding space of a chosen model)

## How to use

There are a number of ways to run this notebook. You can run it locally, or you might wish to run it on a server solution such as google colab (and to avail yourself of their GPUs).

### Running Locally
1. `pip install requirements.txt`
2. `python3 demo.py`

### Running on Colab
This repo depends on Python ~3.10.14, so you will need to reset colab to run an older version. You can select this with the `change runtime` option. This notebook has been tested with the 2025.07 runtime.

<a target="_blank" href="https://colab.research.google.com/github/ashNotKetchup/SAO-Text-Embedding-Playground/blob/main/demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

