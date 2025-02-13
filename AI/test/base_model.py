"""
Test config for BaseModel
"""
import torch

from AI.src.utils import DotDict, ConfigReader
from AI.src.modeling.architectures import build_model, BaseModelOutput


def main() -> None:
    config: DotDict = ConfigReader("../config/base_model_test.json").config
    model: torch.nn.Module = build_model(config)

    with torch.amp.autocast(config.Global.device, torch.float16):
        outs: BaseModelOutput = model(torch.randn(4, 32, 3, 15, 224, 224, device=config.Global.device))
        print(outs[0].shape)
    return None


if __name__ == '__main__':
    main()
