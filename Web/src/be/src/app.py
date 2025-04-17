"""
Main web app module
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "..", ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Web.src.be.src import video , index


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    # return Response status 200
    return {"status": "ok"}

app.include_router(router=index.router)
app.include_router(router=video.router)