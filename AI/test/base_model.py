"""
Test config for BaseModel
"""
import torch

from AI.src.utils.Tracer import LeafModuleAwareTracer
from AI.src.utils import DotDict, ConfigReader, ModelArchInspector
from AI.src.modeling.architectures import build_model, BaseModelOutput


def main() -> None:
    inspect: bool = True
    config: DotDict = ConfigReader("../config/base_model_test.json").config

    model: torch.nn.Module = build_model(config)
    with torch.amp.autocast(config.Global.device, torch.float16):
        x: torch.Tensor = torch.randn(1, 32, 3, 15, 224, 224, device=config.Global.device)

        if inspect:
            ModelArchInspector(
                model, x.shape,
                depth=2,
                device="cuda",
                verbose=1,
                mode="eval",
            )()

        outs: BaseModelOutput = model(x)
        print("Model out", outs[0].shape)
    return None


if __name__ == '__main__':
    main()
