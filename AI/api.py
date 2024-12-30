"""
API for model will be defined here.
"""
import torch
import uvicorn
from fastapi import FastAPI, Response

app = FastAPI()


@app.get("/health-check")
def check_health():
    return {
        "exit code": Response(status_code=200).status_code,
    }


@app.get("/cuda-check")
def check_cuda():
    return {
        "Cuda status": torch.cuda.is_available(),
    }


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=6968)
    return None


if __name__ == '__main__':
    main()