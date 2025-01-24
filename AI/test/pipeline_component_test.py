import gc
import time


import torch
from transformers import CLIPProcessor
from torchvision.models.video import s3d, S3D_Weights


from AI.src.model.CLIP import CLIPModel
from AI.src.model import inception_i3d, MLP
from AI.src.model.TAM import TemporalAggregation

from AI.src.utils.Tracer import LeafModuleAwareTracer
from AI.src.utils.tensor_hook import pack_hook, unpack_hook
from AI.src.utils.ModelArchInspector import ModelArchInspector
from AI.src.utils.pseudo_label_refinement import PseudoLabelRefiner
from AI.src.utils.create_feature_extractor import create_feature_extractor


def test_3D_extract_feature(reduce: str = "first",
                            device: str = "cuda"
                            ) -> None:
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

    video: torch.Tensor = torch.rand((80, 3, 16, 224, 224), device=device, dtype=torch.float16)
    tracer: LeafModuleAwareTracer = LeafModuleAwareTracer()

    with (torch.autocast(device_type=device, dtype=torch.float16)):
        for model_name, return_nodes, weights in zip(
                ["s3d", "i3d"],
                [{"avgpool": "features"}, {"avg_pool": "features"}],
                [S3D_Weights.DEFAULT, "../weights/I3D/rgb.pt"]
        ):
            print("Current model: {}".format(model_name))

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

            feature_extractor: torch.fx.graph_module.GraphModule = create_feature_extractor(model, return_nodes).eval()
            feature_extractor = feature_extractor.to(device)

            # Forward with offloading. Able run with large input
            # with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            #     features: torch.Tensor = feature_extractor(video)["features"]  # [B, C, 1, 1, 1] due to 3DConv
            #     features = features.squeeze()

            # Normal forward
            with torch.no_grad():
                features: torch.Tensor = feature_extractor(video)["features"]  # [B, C, 1, 1, 1] due to 3DConv

            features = features.squeeze(-1).squeeze(-1)

            if reduce == "mean":
                features = features.mean(dim=-1)
            else:
                features = features[..., 0]

            print(f"""Test feature extractor
Feature extractor: {model.__class__.__name__}
Input: {video.shape}
Output: {features.shape}
""")
            gc.collect()
            torch.cuda.empty_cache()
    return None


def test_2D_extract_feature(device: str = "cuda") -> None:
    """
    :return: extracted features
    More feature extractors can be found at: https://github.com/v-iashin/video_features/tree/master
    """
    video_frames: torch.Tensor = torch.rand((128, 3, 224, 224), device=device, dtype=torch.float16)

    model: torch.nn.Module = CLIPModel.from_pretrained("../weights/CLIP/vit-base-patch16/", use_safetensors=True)
    preprocessor = CLIPProcessor.from_pretrained("../weights/CLIP/vit-base-patch16", do_rescale=False)

    tracer: LeafModuleAwareTracer = LeafModuleAwareTracer()
    with (torch.autocast(device_type=device, dtype=torch.float16)):
        # inputs: dict = preprocessor(
        #     text=["a photo of a dog"], images=video, return_tensors="pt", padding=True
        # )
        # ModelArchInspector(
        #     model, None, inputs.pop("input_ids"),
        #     depth=5,
        #     device="cuda",
        #     verbose=1,
        #     mode="eval",
        #     **inputs
        # )()

        # graph: torch.fx.graph.Graph = tracer.trace(model, concrete_args={"return_loss": None, "return_dict": None})
        # graph.print_tabular()

        feature_extractor: torch.fx.graph_module.GraphModule = create_feature_extractor(
            model,
            {"visual_projection": "features"},
            concrete_args={"return_loss": None, "return_dict": None}
        ).eval()
        feature_extractor = feature_extractor.to(device)

        # Forwarding model
        inputs: dict = preprocessor(
            text=["a photo of a cat", "a photo of a dog"], images=video_frames, return_tensors="pt", padding=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        features = feature_extractor(**inputs)["features"]  # [B, D] due to 3DConv

        print(f"""Test feature extractor
Feature extractor: {model.__class__.__name__}
Input: {video_frames.shape}
Output: {features.shape}
""")
        gc.collect()
        torch.cuda.empty_cache()


def test_mlp(device: str = "cuda") -> None:
    seq_len: int = 32
    embed_dim: int = 1024

    mlp: MLP = MLP(
        embed_dim,
        [512, 512],
        512,
        True,
        0.5,
        torch.nn.ReLU()
    ).to(device)

    features = torch.rand((1, seq_len, embed_dim), device=device)

    with torch.autocast(device, torch.float16):
        outputs: torch.Tensor = mlp(features)

    print(f"""Test MLP result:
Input shape: {features.shape}
Output shape: {outputs.shape}
""")
    gc.collect()
    torch.cuda.empty_cache()
    return None


def test_TAM(device: str = "cuda") -> None:
    batch_size = 10
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


def test_pseudo_label_refiner(device: str = "cuda") -> None:
    batch_size = 1
    seq_len = 32
    embed_dim = 512
    mask_threshold = .9

    mlp = MLP(
        embed_dim,
        [512, 32],
        1,
        True,
        0.5,
        torch.nn.ReLU(),
        torch.nn.Sigmoid()
    ).to(device)
    print(mlp)
    pseudo_label_refiner: PseudoLabelRefiner = PseudoLabelRefiner()

    # Output shape from TAM [batch_size, seq_len, embed_dim]
    x: torch.Tensor = torch.rand((batch_size, seq_len, embed_dim), device=device)

    with torch.amp.autocast(device, torch.float16):
        # [batch_size, seq_len, num_classes=1]
        anomalous_scores: torch.Tensor = mlp(x)
        anomalous_scores = pseudo_label_refiner(anomalous_scores)

        mask = torch.where(anomalous_scores > mask_threshold, 1, 0)

        print(f"""Test segment-level prediction/ soft pseudo-labels
{mask.squeeze()}""")


def main() -> None:
    test_3D_extract_feature()
    test_2D_extract_feature()
    test_mlp()
    test_TAM()
    test_pseudo_label_refiner()
    return None


if __name__ == '__main__':
    main()
