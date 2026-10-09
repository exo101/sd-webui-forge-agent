# HuggingFace Guess
A simple tool to guess an HuggingFace repo URL from a state dict.

> The main model detection logics are extracted from **Diffusers** and stolen from **ComfyUI**.

<br>

- This repo does almost the same thing as the following code, but a bit stronger and more robust.

```py
from diffusers.loaders.single_file_utils import fetch_diffusers_config
```

- The following code will print `runwayml/stable-diffusion-v1-5`

```py
import safetensors.torch as sf
import huggingface_guess


state_dict = sf.load_file("./realisticVisionV51_v51VAE.safetensors")
repo_name = huggingface_guess.guess_repo_name(state_dict)
print(repo_name)
```

<br>

Then you can download (or prefetch configs) from HuggingFace to instantiate models and load weights.
