import torch
from diffusers import StableDiffusionPipeline
from IPython.display import display
from PIL import Image

# Replace the model version with your required version if needed
pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
)

device = torch.device("cpu")

# Running the inference on GPU with cuda enabled
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")

print(f'Running on: {device}')

pipeline = pipeline.to(device)

prompt = "A cat with glass"

image = pipeline(prompt=prompt).images[0]

display(image)