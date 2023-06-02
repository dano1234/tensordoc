import os
import time
import uuid


from huggingface_hub import login

huggingface_key = os.environ.get("huggingface_key")
print("huggingface_key")
print(huggingface_key)
login(token='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')

# print("checking CUDA")
# print(f"Is CUDA available: {torch.cuda.is_available()}")
# # True
# print(
#     f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# from fastapi import FastAPI
# from fastapi.exceptions import HTTPException
# from fastapi.responses import FileResponse
# from PIL import Image
# from pydantic import BaseModel, Field
# from stable_diffusion_tf.stable_diffusion import StableDiffusion
# from tensorflow import keras

# height = int(os.environ.get("WIDTH", 512))
# width = int(os.environ.get("WIDTH", 512))
# mixed_precision = os.environ.get("MIXED_PRECISION", "no") == "yes"

# if mixed_precision:
#     keras.mixed_precision.set_global_policy("mixed_float16")

# generator = StableDiffusion(img_height=height, img_width=width, jit_compile=False)




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


###########################

device = "cuda"

repo_id = "stabilityai/stable-diffusion-2-base"

class MyPipelineTxt2Img(StableDiffusionPipeline):

  def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt,prompt_embeds,negative_prompt_embeds):
    te = super()._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt,prompt_embeds,negative_prompt_embeds)
    self.trans_text_embeddings = te
    if self.use_supplied_text_embeddings  ==  True:
      te = self.supplied_text_embeddings
    else:
      te = super()._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt,prompt_embeds,negative_prompt_embeds)
      self.trans_text_embeddings = te
    self.use_supplied_text_embeddings =  False
    return te

  def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator,latents):
    if self.use_supplied_prepared_latents  ==  True:
      prepared_latents = self.supplied_prepared_latents
    else:
      prepared_latents = super().prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator,latents)
    self.trans_prepared_latents = prepared_latents
    self.use_supplied_prepared_latents =  False
    return prepared_latents

  def decode_latents(self,latents):
    self.trans_latents = latents #json_str;
    image = super().decode_latents(latents)
    return image

pipeTxt2Img = MyPipelineTxt2Img.from_pretrained(
    repo_id, 
    revision="fp16", 
    torch_dtype=torch.float16, 
    use_auth_token=True
    )
pipeTxt2Img.use_supplied_text_embeddings  = False
pipeTxt2Img.scheduler = DPMSolverMultistepScheduler.from_config(pipeTxt2Img.scheduler.config)
pipeTxt2Img.to(device)


repo_id = "stabilityai/stable-diffusion-2-inpainting"

class MyPipelineInpainting(StableDiffusionInpaintPipeline):  
  def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
    te = super()._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)
    self.trans_text_embeddings = te
    return te
  def decode_latents(self,latents):
    self.trans_latents = latents #json_str;
    image = super().decode_latents(latents)
    return image
pipeInpainting = MyPipelineInpainting.from_pretrained(  #repo_id,**pipeTxt2Img.components)
    repo_id,
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
    )
pipeInpainting.scheduler = DPMSolverMultistepScheduler.from_config(pipeInpainting.scheduler.config)
pipeInpainting.to(device)

repo_id = "stabilityai/stable-diffusion-2-base"

class MyPipelineImg2Img(StableDiffusionImg2ImgPipeline):  
  def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt,prompt_embeds,negative_prompt_embeds):
    te = super()._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt,prompt_embeds,negative_prompt_embeds)
    self.trans_text_embeddings = te
    return te
    
  def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
    if self.use_supplied_prepared_latents  ==  True:
      prepared_latents = self.supplied_prepared_latents
    else:
      prepared_latents = super().prepare_latents( image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None)
    self.trans_prepared_latents = prepared_latents
    self.use_supplied_prepared_latents =  False
    return prepared_latents

  def decode_latents(self,latents):
    self.trans_latents = latents #json_str;
    image = super().decode_latents(latents)
    return image



pipeImg2Img = MyPipelineImg2Img.from_pretrained(  repo_id,**pipeTxt2Img.components)
pipeImg2Img.use_supplied_prepared_latents = False
pipeImg2Img.scheduler = LMSDiscreteScheduler.from_config(pipeImg2Img.scheduler.config)
pipeImg2Img.to(device)

#######################################

from pydantic.errors import NoneIsNotAllowedError
from fastapi import FastAPI
import nest_asyncio
import json
from pyngrok import ngrok
import uvicorn
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class OneImage(BaseModel):
  prompt: str
  image: str

class Item(BaseModel):
    prompt: str
    #description: Union[str, None] = None
    #price: float
    #tax: Union[float, None] = None

class TwoLatents(BaseModel):
  firstLatents: str
  secondLatents: str
  percent: float

class StringArrayInput(BaseModel):
    arrayOfStrings: List[str]

class TwoPrompts(BaseModel):
  firstPrompt: str
  secondPrompt: str
  percent: float

class InpaintParams(BaseModel):
  prompt: str
  image: str
  mask_image: str

app = FastAPI()

origins = [
    "*",
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get('/index')
async def home():
  return "Hello World"


@app.post("/generateIt/")
async def create_item(item: Item):
  print("-------------GenerateIt")
  promptFromClient = item.prompt;
  pipeTxt2Img.use_supplied_prepared_latents = False
  images = pipeTxt2Img(prompt=promptFromClient, num_inference_steps=25).images
  b64Image = encodeb64(images[0])
  trans_latents = pipeTxt2Img.trans_latents.cpu().numpy()
  trans_latents  = trans_latents.tolist()
  latents_json_str = json.dumps(trans_latents )
  trans_text_embeddings = pipeTxt2Img.trans_text_embeddings.tolist()
  text_embeddings_json_str = json.dumps(trans_text_embeddings )

  d = {
    "prompt": promptFromClient, 
    "b64Image": b64Image, 
    "latents": latents_json_str,
    "text_embeddings" : text_embeddings_json_str
  }
  #return jsonify(d)
  json_compatible_item_data = jsonable_encoder(d)
  return JSONResponse(content=json_compatible_item_data)

@app.post("/generateInpaint/")
async def create_item(inpaintParams: InpaintParams):
  print("-------------generateInpaint")
  promptFromClient = inpaintParams.prompt
  imageFromClient = inpaintParams.image
  mask_imageFromClient = inpaintParams.mask_image
  PILImage =     Image.open(BytesIO(base64.b64decode(imageFromClient)))
  PILMaskImage = Image.open(BytesIO(base64.b64decode(mask_imageFromClient)))
  guidance_scale=7.5
  num_samples = 1
  generator = torch.Generator(device="cuda").manual_seed(0) # change the seed to get different results

  images = pipeInpainting(
    prompt=promptFromClient,
    image=PILImage,
    mask_image=PILMaskImage,
    guidance_scale=guidance_scale,
    generator=generator,
    num_images_per_prompt=num_samples
    ).images

  b64Image = encodeb64(images[0])
  trans_latents = pipeInpainting.trans_latents.cpu().numpy()
  trans_latents  = trans_latents.tolist()
  latents_json_str = json.dumps(trans_latents )
  trans_text_embeddings = pipeInpainting.trans_text_embeddings.tolist()
  text_embeddings_json_str = json.dumps(trans_text_embeddings )


  d = {
    "name": promptFromClient, 
    "b64Image": b64Image, 
    "latents": latents_json_str,
    "text_embeddings":text_embeddings_json_str
  }

  json_compatible_item_data = jsonable_encoder(d)
  return JSONResponse(content=json_compatible_item_data)


@app.post("/findBetweenPrompts/")
async def create_item(twoPrompts: TwoPrompts):
  print("-------------findBetweenPrompts")
  start_text = twoPrompts.firstPrompt;
  end_text = twoPrompts.secondPrompt;
  percent = twoPrompts.percent;
  start_embedding = pipeTxt2Img._encode_prompt(start_text, device, 1,  True, None, None, None)
  end_embedding = pipeTxt2Img._encode_prompt(end_text, device, 1, True, None,None,None)
  #start_embedding = start_embedding[0].cpu().detach().numpy()
  #end_embedding = end_embedding[0].cpu().detach().numpy()
  pipeTxt2Img.use_supplied_prepared_latents = False
  pipeTxt2Img.use_supplied_text_embeddings = True
  with torch.no_grad():  #??
    pipeTxt2Img.supplied_text_embeddings = interpolate(percent, start_embedding, end_embedding)
  images = pipeTxt2Img(prompt="", num_inference_steps=50).images
  b64Image = encodeb64(images[0])
  trans_latents = pipeTxt2Img.trans_latents.cpu().numpy()
  trans_latents  = trans_latents.tolist()
  latents_json_str = json.dumps(trans_latents )
  trans_text_embeddings = pipeTxt2Img.trans_text_embeddings.tolist()
  text_embeddings_json_str = json.dumps(trans_text_embeddings )
  d = {
    "prompt1": start_text, 
    "prompt2": end_text,
    "b64Image": b64Image, 
    "latents": latents_json_str,
    "text_embeddings" : text_embeddings_json_str
  }
  json_compatible_item_data = jsonable_encoder(d)
  return JSONResponse(content=json_compatible_item_data)


@app.post("/findBetweenPreparedLatents/")
async def create_item(twoLatents: TwoLatents):
  print("-------------findBetweenPreparedLatents")
  start_latents = twoLatents.firstLatents;
  end_latents = twoLatents.secondLatents;
  percent = twoLatents.percent;
  start_latents = np.array(json.loads(start_latents)) 
  end_latents = np.array(json.loads(end_latents)) 
  supplied_prepared_latents = interpolate(percent, start_latents, end_latents)
  pipeImg2Img.supplied_prepared_latents =torch.from_numpy(supplied_prepared_latents).to(device=device,dtype=torch.float16)
  pipeImg2Img.use_supplied_prepared_latents = True;
  # with torch.no_grad():
  #  pipeTxt2Img.supplied_prepared_latents = interpolate(percent,  start_latents, end_latents )
  generator = torch.Generator(device=device).manual_seed(1024)
  img = Image.new("RGB", (512, 512), (255, 255, 255))
  images = pipeImg2Img(prompt="", num_inference_steps=50,  image= img, strength=0.2, guidance_scale=0.1 , generator=generator).images

  #images = pipeImg2Img(prompt="", num_inference_steps=25).images
  #images = pipeImg2Img(prompt="", num_inference_steps=25,  image= PILImage, strength=0.5, guidance_scale=7.5 , generator=generator).images
  b64Image = encodeb64(images[0])
  trans_prepared_latents = pipeImg2Img.trans_prepared_latents.cpu().numpy()
  trans_prepared_latents  = trans_prepared_latents.tolist()
  prepared_latents_json_str = json.dumps(trans_prepared_latents)
  trans_text_embeddings = pipeImg2Img.trans_text_embeddings.tolist()
  text_embeddings_json_str = json.dumps(trans_text_embeddings )
  d = {
    "b64Image": b64Image, 
    "prepared_latents": prepared_latents_json_str ,
    "text_embeddings" : text_embeddings_json_str
  }
  json_compatible_item_data = jsonable_encoder(d)
  return JSONResponse(content=json_compatible_item_data)

@app.post("/findBetweenLatents/")
async def create_item(twoLatents: TwoLatents):
  print("-------------findBetweenLatents")
  start_latents = twoLatents.firstLatents;
  end_latents = twoLatents.secondLatents;
  percent = twoLatents.percent;
  start_latents = np.array(json.loads(start_latents)) 
  end_latents = np.array(json.loads(end_latents)) 
  supplied_latents = interpolate(percent, start_latents, end_latents)
  supplied_latents =torch.from_numpy(supplied_latents).to(device=device,dtype=torch.float16)
  with torch.no_grad():
    images = pipeImg2Img.decode_latents(supplied_latents)
  b64Image = encodeb64(images[0])
  trans_latents = supplied_latents.cpu().numpy()
  trans_latents  = trans_latents.tolist()
  json_latents = json.dumps(trans_latents )
  d = {
    "b64Image": b64Image, 
    "latents": json_latents
  }

  json_compatible_item_data = jsonable_encoder(d)
  return JSONResponse(content=json_compatible_item_data)


@app.post("/findInterpolationBetweenAllLatents/")
async def create_item(stringArrayInput: StringArrayInput):
  arrayOfPrompts = stringArrayInput.arrayOfStrings
  print("-------------findBetweenAllLatents")
  previousLatent = pipeTxt2Img._encode_prompt(arrayOfPrompts[0], device, 1,  True, None, None, None)
  num_latents= len(arrayOfPrompts)
  for i in range(1, num_latents): 
        newLatent = pipeTxt2Img._encode_prompt(arrayOfPrompts[i], device, 1,  True, None, None, None)
        with torch.no_grad():  #??
          previousLatent = interpolate(0.5,previousLatent,newLatent)

  #start_embedding = start_embedding[0].cpu().detach().numpy()
  #end_embedding = end_embedding[0].cpu().detach().numpy()
  pipeTxt2Img.use_supplied_text_embeddings = True;
  pipeTxt2Img.supplied_text_embeddings = previousLatent
  images = pipeTxt2Img(prompt="", num_inference_steps=40).images
  b64Image = encodeb64(images[0])
  trans_latents = pipeTxt2Img.trans_latents.cpu().numpy()
  trans_latents  = trans_latents.tolist()
  latents_json_str = json.dumps(trans_latents )
  trans_text_embeddings = pipeTxt2Img.trans_text_embeddings.tolist()
  text_embeddings_json_str = json.dumps(trans_text_embeddings )
  d = {
    "prompt1": "nothing yet", 
    "prompt2": "nothing yet",
    "b64Image": b64Image, 
    "latents": latents_json_str,
    "text_embeddings" : text_embeddings_json_str
  }
  json_compatible_item_data = jsonable_encoder(d)
  return JSONResponse(content=json_compatible_item_data)

  # supplied_latents = interpolateBetweenAllLatents(arrayOfStrings)
  # supplied_latents =torch.from_numpy(supplied_latents).to(device=device,dtype=torch.float16)
  # with torch.no_grad():
  #   images = pipeImg2Img.decode_latents(supplied_latents)
  # b64Image = encodeb64(images[0])
  # trans_latents = supplied_latents.cpu().numpy()
  # trans_latents  = trans_latents.tolist()
  # json_latents = json.dumps(trans_latents )
  # d = {
  #   "b64Image": b64Image, 
  #   "latents": json_latents
  # }

  # json_compatible_item_data = jsonable_encoder(d)
  # return JSONResponse(content=json_compatible_item_data)


# @app.post("/findBetweenImages/")
# async def create_item(twoLatents: TwoLatents):

#   start_latents = twoLatents.firstLatents;
#   end_latents = twoLatents.secondLatents;
#   percent = twoLatents.percent;
#   start_latents = np.array(json.loads(start_latents)) 
#   end_latents = np.array(json.loads(end_latents)) 
#   supplied_latents = interpolate(percent, start_latents, end_latents)
#   supplied_latents = torch.from_numpy(supplied_latents).to(device=device) #, dtype=torch.float
#   b64Image = encodeb64(image)
#   trans_latents = supplied_latents.cpu().numpy()
#   trans_latents  = trans_latents.tolist()
#   json_latents = json.dumps(trans_latents )
#   d = {
#     "b64Image": b64Image, 
#     "latents": json_latents
#   }

#   json_compatible_item_data = jsonable_encoder(d)
#   return JSONResponse(content=json_compatible_item_data)



@app.post("/getImg2Img/")
async def create_item(oneImage: OneImage):
  print("-------------getImg2Img")
  promptFromClient = oneImage.prompt;
  PILImage = Image.open(BytesIO(base64.b64decode(oneImage.image)))
  generator = torch.Generator(device=device).manual_seed(1024)
  image = pipeImg2Img(prompt=promptFromClient, num_inference_steps=20,  image= PILImage, strength=0.3, guidance_scale=0.1 , generator=generator).images[0]
  b64Image = encodeb64(image)
  trans_prepared_latents = pipeImg2Img.trans_prepared_latents.cpu()
  trans_prepared_latents =trans_prepared_latents.numpy()
  trans_prepared_latents  = trans_prepared_latents.tolist()
  latents_json_str = json.dumps(trans_prepared_latents )
  #trans_text_embeddings = pipeImg2Img.trans_text_embeddings.tolist()
  #text_embeddings_json_str = json.dumps(trans_text_embeddings )

  d = {
    "prompt": promptFromClient, 
    "b64Image": b64Image, 
    "prepared_latents": latents_json_str,
    #"text_embeddings" : text_embeddings_json_str
  }
  #return jsonify(d)
  json_compatible_item_data = jsonable_encoder(d)
  return JSONResponse(content=json_compatible_item_data)

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)



# app = FastAPI(title="Stable Diffusion API")

# class GenerationRequest(BaseModel):
#     prompt: str = Field(..., title="Input prompt", description="Input prompt to be rendered")
#     scale: float = Field(default=7.5, title="Scale", description="Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
#     steps: int = Field(default=50, title="Steps", description="Number of dim sampling steps")
#     seed: int = Field(default=None, title="Seed", description="Optionally specify a seed for reproduceable results")


# class GenerationResult(BaseModel):
#     download_id: str = Field(..., title="Download ID", description="Identifier to download the generated image")
#     time: float = Field(..., title="Time", description="Total duration of generating this image")


# @app.get("/")
# def home():
#     return {"message": "See /docs for documentation"}

# @app.post("/generatee", response_model=GenerationResult)
# def generate(req: GenerationRequest):
#     start = time.time()
#     id = str(uuid.uuid4())
#     img = generator.generate(req.prompt, num_steps=req.steps, unconditional_guidance_scale=req.scale, temperature=1, batch_size=1, seed=req.seed)
#     path = os.path.join("/app/data", f"{id}.png")
#     Image.fromarray(img[0]).save(path)
#     alapsed = time.time() - start
    
#     return GenerationResult(download_id=id, time=alapsed)

# @app.get("/download/{id}", responses={200: {"description": "Image with provided ID", "content": {"image/png" : {"example": "No example available."}}}, 404: {"description": "Image not found"}})
# async def download(id: str):
#     path = os.path.join("/app/data", f"{id}.png")
#     if os.path.exists(path):
#         return FileResponse(path, media_type="image/png", filename=path.split(os.path.sep)[-1])
#     else:
#         raise HTTPException(404, detail="No such file")
