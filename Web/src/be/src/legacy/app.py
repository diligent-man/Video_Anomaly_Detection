"""
Main web app module
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "Web", "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Web.src.be.src import video
from src.api import index, video


# from .Frontend.legacy import (
#     page_1, page_2 
# )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thêm endpoint health check
@app.get("/health")
def health_check():
    return {"status": "ok"}

# app = gr.mount_gradio_app(app=app, blocks=page_1, path="/gradio_p1")
# app = gr.mount_gradio_app(app=app, blocks=page_2, path="/gradio_p2")

app.include_router(router=index.router)
app.include_router(router=video.router)
