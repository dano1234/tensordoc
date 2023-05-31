import os
import time
import uuid

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field
from stable_diffusion_tf.stable_diffusion import StableDiffusion
from tensorflow import keras

height = int(os.environ.get("WIDTH", 512))
width = int(os.environ.get("WIDTH", 512))
mixed_precision = os.environ.get("MIXED_PRECISION", "no") == "yes"

if mixed_precision:
    keras.mixed_precision.set_global_policy("mixed_float16")

generator = StableDiffusion(img_height=height, img_width=width, jit_compile=False)




import inspect
from typing import List, Optional, Union

import numpy as np
import torch

import PIL
#import gradio as gr
#from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline,  DPMSolverMultistepScheduler
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler
)

###############



from PIL import Image
from io import BytesIO
import base64
import skimage
import numpy 

def encodeb64(image) -> str:

    # convert image to bytes
    with BytesIO() as output_bytes:
        PIL_image = Image.fromarray(skimage.img_as_ubyte(image))
        PIL_image.save(output_bytes, 'JPEG') # Note JPG is not a vaild type here
        bytes_data = output_bytes.getvalue()

    # encode bytes to base64 string
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return base64_str

def interpolateBetweenAllLatents(latents):
    previousLatent = np.array(latents[0])  # Initialize a vector of zeros with the same length as the input vectors

 
    num_latents= len(latents)
    
    for i in range(1, num_latents): 
        newLatent= np.array(json.loads(latents[i]))
        previousLatent = interpolate(0.5,previousLatent,newLatent)

    return previousLatent

def interpolate(t, v0, v1, DOT_THRESHOLD=0.9995):
  """Helper function to (spherically) interpolate two arrays v1 v2.
  Taken from: https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355
  """
  inputs_are_torch = False
  if not isinstance(v0, np.ndarray):
    inputs_are_torch = True
    input_device = v0.device
    v0 = v0.cpu().numpy()
    v1 = v1.cpu().numpy()
    #print("came in on device")
  #else:
    #print("came in off device")

  dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
  if np.abs(dot) > DOT_THRESHOLD:
    v2 = (1 - t) * v0 + t * v1
  else:
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0 + s1 * v1

  if inputs_are_torch:
    v2 = torch.from_numpy(v2).to(input_device)
    #print("were returned to device")

  return v2

def generateFromOnlyLatent(pipeline, latents):
  latents = 1 / 0.18215 * latents
  image = pipeline.vae.decode(latents).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
  image = image.cpu().permute(0, 2, 3, 1).float().numpy()
  return image











app = FastAPI(title="Stable Diffusion API")





class GenerationRequest(BaseModel):
    prompt: str = Field(..., title="Input prompt", description="Input prompt to be rendered")
    scale: float = Field(default=7.5, title="Scale", description="Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    steps: int = Field(default=50, title="Steps", description="Number of dim sampling steps")
    seed: int = Field(default=None, title="Seed", description="Optionally specify a seed for reproduceable results")


class GenerationResult(BaseModel):
    download_id: str = Field(..., title="Download ID", description="Identifier to download the generated image")
    time: float = Field(..., title="Time", description="Total duration of generating this image")


@app.get("/")
def home():
    return {"message": "See /docs for documentation"}

@app.post("/generatee", response_model=GenerationResult)
def generate(req: GenerationRequest):
    start = time.time()
    id = str(uuid.uuid4())
    img = generator.generate(req.prompt, num_steps=req.steps, unconditional_guidance_scale=req.scale, temperature=1, batch_size=1, seed=req.seed)
    path = os.path.join("/app/data", f"{id}.png")
    Image.fromarray(img[0]).save(path)
    alapsed = time.time() - start
    
    return GenerationResult(download_id=id, time=alapsed)

@app.get("/download/{id}", responses={200: {"description": "Image with provided ID", "content": {"image/png" : {"example": "No example available."}}}, 404: {"description": "Image not found"}})
async def download(id: str):
    path = os.path.join("/app/data", f"{id}.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png", filename=path.split(os.path.sep)[-1])
    else:
        raise HTTPException(404, detail="No such file")
