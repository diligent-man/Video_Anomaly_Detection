import os
import tempfile
from functools import partial

import torch
from torchvision.models.video import s3d


from AI.src.model import inception_i3d, MLP
from AI.src.utils.Tracer import LeafModuleAwareTracer
from AI.src.utils.ModelArchInspector import ModelArchInspector
from AI.src.utils.create_feature_extractor import create_feature_extractor
from AI.src.utils.tensor_hook import pack_hook, unpack_hook

from torch.nn import MultiheadAttention

from torchvision.models.video.s3d import S3D_Weights


def extract_feature(reduce: str = "first") -> torch.Tensor:
    """
    :param reduce: how to get final extracted features. 3D-CNN-related computations will return more than 1 timeframe
                   if T > 13. We can choose to take first time frame or mean all timeframes as extracted features.
                   ["first", "mean"]
    :return: extracted features
    """
    # Divide 1 video into 32 non-overlapping segments
    # Suppose fps=30, video length: T(s) => Total frames: 30*T frames
    # Shape: 32, 30 * T//32, 3, 224, 224 => (STCHW)
    # Note: Minimum timestep=13
    available_feature_extractor = {
        "s3d": s3d,
        "i3d": inception_i3d,
    }

    video: torch.Tensor = torch.rand((32, 3, 80, 224, 224), device="cuda", dtype=torch.float16)
    tracer: LeafModuleAwareTracer = LeafModuleAwareTracer()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for model_name, return_nodes, weights in zip(
            ["s3d", "i3d"],
            [{"avgpool": "features"}, {"avg_pool": "features"}],
            [S3D_Weights.DEFAULT, "./weights/I3D/rgb.pt"]
        ):

            model: torch.nn.Module = available_feature_extractor[model_name](weights=weights)

            # ModelArchInspector(
            #     model, None, video,
            #     depth=2,
            #     device="cuda",
            #     verbose=1,
            #     mode="train"
            # )()

            # graph: torch.fx.graph.Graph = tracer.trace(model)
            # graph.print_tabular()

            feature_extractor: torch.fx.graph_module.GraphModule = create_feature_extractor(model, return_nodes=return_nodes)
            feature_extractor.eval()
            feature_extractor = feature_extractor.to("cuda")

            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                features: torch.Tensor = feature_extractor(video)["features"]  # [B, C, 1, 1, 1] due to 3DConv
                features = features.squeeze()

            if reduce == "mean":
                features = features.mean(dim=-1)
            else:
                features = features[..., 0]
            print(features.shape)
    return None


def main() -> None:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        features = extract_feature()
        # print(features.shape)
        # features = MLP(features.shape[-1]).to("cuda")(features)
        # print(features.shape)
        # print()
    return None


if __name__ == '__main__':
    main()
