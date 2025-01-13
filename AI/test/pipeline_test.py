import torch

from torchvision.models.video import s3d, S3D_Weights

from AI.src.model import inception_i3d, MLP
from AI.src.model.TAM import TemporalAggregation
from AI.src.utils.Tracer import LeafModuleAwareTracer
from AI.src.utils.ModelArchInspector import ModelArchInspector
from AI.src.utils.create_feature_extractor import create_feature_extractor
from AI.src.utils.tensor_hook import pack_hook, unpack_hook
from AI.src.utils.pseudo_label_refinement import PseudoLabelRefiner


def extract_feature(reduce: str = "first") -> None:
    """
    :param reduce: how to get final extracted features. 3D-CNN-related computations will return more than 1 timeframe
                   if T > 13. We can choose to take first time frame or mean all timeframes as extracted features.
                   ["first", "mean"]
    :return: extracted features
    More feature extractors can be found at: https://github.com/v-iashin/video_features/tree/master
    """
    # Divide 1 video into 32 non-overlapping segments
    # Suppose fps=30, video length: T(s) => Total frames: 30*T frames
    # Shape: 32, 30 * T//32, 3, 224, 224 => (STCHW)
    # Note: Minimum timestep=13
    available_feature_extractor = {
        "s3d": s3d,
        "i3d": inception_i3d,
    }

    video: torch.Tensor = torch.rand((32, 3, 26, 224, 224), device="cuda", dtype=torch.float16)
    tracer: LeafModuleAwareTracer = LeafModuleAwareTracer()

    # with torch.autocast(device_type="cuda", dtype=torch.float16):
    for model_name, return_nodes, weights in zip(
            ["s3d", "i3d"],
            [{"avgpool": "features"}, {"avg_pool": "features"}],
            [S3D_Weights.DEFAULT, "../weights/I3D/rgb.pt"]
        ):
            print("Current model: {}".format(model_name))

            if model_name == "s3d":
                continue

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
            feature_extractor = feature_extractor.to("cpu")

            # Forward with offloading. Able run with large input
            # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            #     features: torch.Tensor = feature_extractor(video)["features"]  # [B, C, 1, 1, 1] due to 3DConv
            #     features = features.squeeze()

            # Normal forward
            features: torch.Tensor = feature_extractor(video)["features"]  # [B, C, 1, 1, 1] due to 3DConv
            features = features.squeeze()

            if reduce == "mean":
                features = features.mean(dim=-1)
            else:
                features = features[..., 0]
            print(features.shape)
    return None


def pass_mlp() -> None:
    # with torch.autocast(device_type="cuda", dtype=torch.float16):
    features = torch.rand((32, 1024), device="cuda")  # extract_feature()
    features = MLP(features.shape[-1]).to("cuda")(features)
    print(features.shape)
    return None


def pass_TAM():
    device = "cuda"

    batch_size = 1
    embed_dim = 512
    max_rel_pos = 4
    seq_len = 32

    # Case 1: Single backbone, cross-c2c term will be omitted
    num_backbones = 1
    tam = TemporalAggregation(
        embed_dim,
        num_backbones,
        max_rel_pos,
        *(False, device, torch.float32)
    )
    inputs = torch.rand((num_backbones, batch_size, seq_len, embed_dim), device=device)
    assert tam(inputs).shape == torch.Size([batch_size, seq_len, embed_dim])

    # Case 2: Multiple backbones
    num_backbones = 10
    tam = TemporalAggregation(
        embed_dim,
        num_backbones,
        max_rel_pos,
        *(False, device, torch.float32)
    )
    inputs = torch.rand((num_backbones, batch_size, seq_len, embed_dim), device=device)
    assert tam(inputs).shape == torch.Size([batch_size, seq_len, embed_dim])
    return None


def main() -> None:
    # extract_feature()
    # pass_mlp()
    pass_TAM()
    return None


if __name__ == '__main__':
    main()
