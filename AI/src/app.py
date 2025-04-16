"""
API for model will be defined here.
"""
import os
import sys

from pathlib import Path
from typing import Dict, Any, Mapping, List
from tempfile import TemporaryDirectory, NamedTemporaryFile
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "."))


import torch

from torch import Tensor
from torch.nn import Module

from fastapi.responses import JSONResponse, Response
from fastapi import FastAPI, UploadFile, File, HTTPException


from AI.src.modeling import build_model
from AI.src.utils import load_config, load_weights, DotDict
from AI.src.utils.inference_ops import extract_frames, load_img


app = FastAPI()

device: str = os.environ.get("DEVICE", "cpu")
overlap_ratio: float = os.environ.get("OVERLAP_RATIO", .5)
T_max: int = os.environ.get("T_MAX", 30)

# load config
config: str = os.environ["CONFIG_PATH"]
config: DotDict = DotDict(load_config(config))

# load checkpoint
weight: str = os.environ["WEIGHT_PATH"]
weight: Mapping[str, Any] = load_weights(weight, weights_only=False)

model: Module = build_model(config)
model.load_state_dict(weight["model"]) if isinstance(weight["model"], dict) else model.load_state_dict(weight["model"].state_dict())
model = model.to(device)


@app.get("/")
def check_health() -> Dict[str, int]:
    return {
        "exit code": Response(status_code=200).status_code,
    }


@app.get("/cuda-check")
def check_cuda() -> Dict[str, Any]:
    return {
        "Cuda status": torch.cuda.is_available(),
        "Cuda version": torch.cuda.get_device_name("cuda")
    }


@app.post("/infer")
async def infer(file: UploadFile = File(...)) -> JSONResponse:
    # Create temporary directory for extracted frames
    try:
        tmp_file = NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_file.write(await file.read())

        with TemporaryDirectory() as tmp_dir:
            extract_frames(tmp_file.name, tmp_dir)
            total_frames: List[str] = [str(f) for f in sorted(Path(tmp_dir).glob("*.jpg"))]

            cum_frames: int = 0
            step_preds: None | Tensor = None
            preds: Tensor = torch.zeros(len(total_frames), dtype=torch.float16, device=device)

            with torch.inference_mode():
                with torch.amp.autocast(enabled=True, dtype=torch.float16, device_type=device):
                    for i in range(len(total_frames)):
                        if i < T_max or cum_frames < T_max:
                            cum_frames += 1
                        else:
                            # step when accumulate sufficient frames
                            inps: Tensor = load_img(total_frames[i - cum_frames: i], torch.float16, device)
                            step_preds: Tensor = model(inps).preds  # (B, S)
                            cum_frames = int(T_max * overlap_ratio) + 1

                        # Last step
                        if i == len(total_frames) - 1:
                            inps: Tensor = load_img(total_frames[i - cum_frames: i], torch.float16, device)
                            step_preds: Tensor = model(inps).preds  # (B, S)

                        if step_preds is not None:
                            step_preds = step_preds.squeeze(0)

                            # First half
                            if preds[i - T_max: i - (T_max // 2)].equal(
                                    torch.zeros_like(preds[i - T_max: i - (T_max // 2)], dtype=preds.dtype,
                                                     device=device)):
                                preds[i - T_max: i - (T_max // 2)] += step_preds
                            else:
                                preds[i - T_max: i - (T_max // 2)] = (preds[
                                                                      i - T_max: i - (T_max // 2)] + step_preds) / 2

                            # Second half
                            if i == len(total_frames) - 1:
                                preds[i - (T_max // 2):] += step_preds
                            else:
                                preds[i - (T_max // 2): i] += step_preds

                            # Reset
                            step_preds = None
        return JSONResponse({"preds": preds.tolist()})
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error processing video: " + str(e))
