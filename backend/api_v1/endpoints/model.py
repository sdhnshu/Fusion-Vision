from typing import Any, List
import torch
from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
import numpy as np
import io
import time
# from sqlalchemy.orm import Session
# from backend.stylegan2.stylegan2 import Generator
from backend.stylegan2.wrappers import StyleGAN2
from fastapi.responses import StreamingResponse
from backend import schemas
# from app.api import deps

router = APIRouter()


def get_model():
    print(time.time())
    model = StyleGAN2(torch.device('cpu'), class_name='ffhq', truncation=0.7, use_w=True)
    model.eval()
    print(time.time())
    return model


@router.post("/generate")
def generate_image(
    model=Depends(get_model)
) -> Any:
    """
    Generate image + 3 neighbours from a 18x512 dim latent
    """
    no_of_seeds = 1
    seed = 0
    # print(time.time(), 'Sampling latent')
    w = model.sample_latent(no_of_seeds, seed=seed).cpu().detach().numpy()
    # print(time.time(), 'Multiplying 18')
    w = [w] * model.get_max_latents()
    # print(time.time(), 'Sampling output')
    out = model.sample_np(w)
    # print(time.time(), 'Sending out')
    img = Image.fromarray((out[0] * 255).astype(np.uint8)).resize((400, 400), Image.LANCZOS)
    return StreamingResponse(io.BytesIO(img.tobytes()), media_type='image/png')
