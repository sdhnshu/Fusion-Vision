from backend.config import settings
from backend.router import api_router
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import uvicorn
import sys


app = FastAPI(title=settings.PROJECT_NAME,
              openapi_url=f"{settings.API_V1_STR}/openapi.json")

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health")
async def health_check():
    return {"msg": "Service is up"}


if __name__ == "__main__":
    if 'dev' in sys.argv:
        uvicorn.run("app:app", host="0.0.0.0", port=8000,
                    reload=True, log_level="info")
    else:
        uvicorn.run("app:app", host="0.0.0.0", port=8000,
                    workers=settings.NO_WORKERS, log_level="info")
