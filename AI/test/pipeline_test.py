import torch
from torchvision.models.video import s3d
from torchvision.models.feature_extraction import create_feature_extractor

from AI.src.model import InceptionI3d, MLP
from AI.src.utils.CustomTracer import CustomTracer
from AI.src.utils.ModelArchInspector import ModelArchInspector


def extract_feature() -> torch.Tensor:
    # Devide 1 video into 32 non-overlapping segments
    # Suppose fps=32, video length: 10s => Total frames: 320 frames
    # Shape: 32, 10, 3, 224, 224 => (STCHW)
    # Note: Minimum timestep=13
    video: torch.Tensor = torch.rand((32, 3, 13, 224, 224), device="cuda", dtype=torch.float16)
    tracer: CustomTracer = CustomTracer()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for model, return_nodes in zip(
            [s3d(weights=None), InceptionI3d()],
            [{"avgpool": "features"}, {"avg_pool": "features"}]
        ):
            model: torch.nn.Module = model.to("cuda")
            model.eval()

            # ModelArchInspector(
            #     model, None, video,
            #     depth=3,
            #     device="cuda",
            #     verbose=1,
            #     mode="train"
            # )()

            # print("Start compile model")
            # model.compile()
            # print("Compilation finished")

            # graph: torch.fx.graph.Graph = tracer.trace(model)
            # print(list(graph.nodes))

            feature_extractor: torch.fx.graph_module.GraphModule = create_feature_extractor(model, return_nodes=return_nodes)
            # print(feature_extractor)

            features: torch.Tensor = feature_extractor(video)["features"]  # [B, C, 1, 1, 1] due to 3DConv
            features = features.squeeze()  # [B, D]
    return features


def main() -> None:
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        features = extract_feature()
        features = MLP(features.shape[-1]).to("cuda")(features)
    return None


if __name__ == '__main__':
    main()
