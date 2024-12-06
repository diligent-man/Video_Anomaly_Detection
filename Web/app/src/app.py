"""
Main web app module
"""
import gradio as gr

from fastapi import FastAPI

from .api import (
    index,
    demo_1,
    demo_2
)

from .gradio import (
    page_1, page_2
)


app = FastAPI()

app = gr.mount_gradio_app(app=app, blocks=page_1, path="/gradio_p1")
app = gr.mount_gradio_app(app=app, blocks=page_2, path="/gradio_p2")

app.include_router(router=index.router, prefix="")
app.include_router(router=demo_1.router, prefix="/demo_api_1")
app.include_router(router=demo_2.router, prefix="/demo_api_2")

