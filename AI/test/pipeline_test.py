import os
import tempfile


import torch
from torchvision.models.video import s3d
from torchvision.models.feature_extraction import create_feature_extractor

from AI.src.model import inception_i3d, MLP
from AI.src.utils.CustomTracer import CustomTracer
from AI.src.utils.ModelArchInspector import ModelArchInspector

from torch.nn import MultiheadAttention


def pack_hook(tensor: torch.Tensor) -> tempfile.NamedTemporaryFile:
    print(tensor.shape)
    tmp_file: tempfile.NamedTemporaryFile = tempfile.NamedTemporaryFile(delete=True)
    torch.save(tensor, tmp_file.name)
    return tmp_file


def unpack_hook(tmp_file: tempfile.NamedTemporaryFile) -> torch.Tensor:
    intermediary_tensor: torch.Tensor = torch.load(tmp_file.name, weights_only=True)
    return intermediary_tensor


def extract_feature() -> torch.Tensor:
    # Divide 1 video into 32 non-overlapping segments
    # Suppose fps=30, video length: T(s) => Total frames: 30*T frames
    # Shape: 32, 30*T//32, 3, 224, 224 => (STCHW)
    # Note: Minimum timestep=13
    available_feature_extractor = {
        "s3d": s3d,
        "i3d": inception_i3d
    }

    # Offload during training
    video: torch.Tensor = torch.rand((3, 13, 224, 224), device="cuda", dtype=torch.float16)
    tracer: CustomTracer = CustomTracer()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for model_name, return_nodes, weights in zip(
            ["s3d", "i3d"],
            [{"avgpool": "features"}, {"avg_pool": "features"}],
            [None, "./weights/I3D/rgb.pt"]
        ):

            model: torch.nn.Module = available_feature_extractor[model_name](weights=weights)
            # device_map: OrderedDict = infer_auto_device_map(model, {0: "2GB", "cpu": "40GB"}, verbose=False)
            # model = dispatch_model(model, device_map)

            ModelArchInspector(
                model, None, video,
                depth=2,
                device="cuda",
                verbose=1,
                mode="train"
            )()

            # print("Start compile model")
            # model.compile()
            # print("Compilation finished")

            # graph: torch.fx.graph.Graph = tracer.trace(model)
            # pp(list(graph.nodes))

            # feature_extractor: torch.fx.graph_module.GraphModule = create_feature_extractor(model, return_nodes=return_nodes)
            # feature_extractor.eval()
            # feature_extractor = feature_extractor.to("cuda")
            #
            # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            #     features: torch.Tensor = feature_extractor(video)["features"]  # [B, C, 1, 1, 1] due to 3DConv
            #     # features = features.squeeze()  # [B, D]
            # print(features.shape)
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
