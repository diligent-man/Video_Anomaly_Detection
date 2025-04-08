"""
Main web app module
"""
import gradio as gr
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from .api import (
    index,
    video,
)

from .Frontend.legacy import (
    page_1, page_2 
)



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app = gr.mount_gradio_app(app=app, blocks=page_1, path="/gradio_p1")
app = gr.mount_gradio_app(app=app, blocks=page_2, path="/gradio_p2")

app.include_router(router=index.router)
app.include_router(router=video.router)



