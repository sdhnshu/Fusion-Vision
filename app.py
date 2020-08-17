from fastapi import FastAPI
import uvicorn
import os
import sys
import torch
app = FastAPI()


# for setup
# Download pretrained stylegan2 models, PCA vectors from ganspace trainings and stuff


@app.get("/health")
async def health_check():
    return {"msg": "Service is up"}


@app.get("/")
async def yo():
    env = os.environ.get('DATABASE_SERVICE_NAME')
    return {"msg": f"This is groot, {env}",
            "python-version": f"{sys.version}",
            'torch': f'{torch.__version__}'}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
