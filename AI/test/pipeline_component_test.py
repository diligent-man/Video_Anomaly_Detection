import gc


import torch
from transformers import CLIPProcessor
from torchvision.models.video import s3d, S3D_Weights

from AI.src.model.MLP import MLP
from AI.src.model.CLIP import CLIPModel
from AI.src.model.TA import TAModel, TAM
from AI.src.model.InceptionI3D import inception_i3d

from AI.src.utils.Tracer import LeafModuleAwareTracer
from AI.src.utils.tensor_hook import pack_hook, unpack_hook
from AI.src.utils.ModelArchInspector import ModelArchInspector
from AI.src.postprocessing import PseudoLabelRefiner
from AI.src.utils.create_feature_extractor import create_feature_extractor








def test_mlp(device: str = "cuda") -> None:
    seq_len: int = 32
    embed_dim: int = 1024

    mlp: MLP = MLP(
        embed_dim,
        [512, 512],
        512,
        True,
        0.1,
        hidden_activation="LeakyReLU",
        out_activation="Sigmoid",
        layer_order="fc->act->drop"
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

    seq_len = 32
    hidden_dim = 1024
    embed_dim = 1024

    max_rel_pos = 4

    # Case 1: Single backbone, cross-c2c term will be omitted
    num_backbones = 1
    tam = TAM(
        num_backbones,
        True,
        max_rel_pos,
        embed_dim
    )

    tam = tam.to(device)
    inputs = torch.rand((num_backbones, batch_size, seq_len, hidden_dim), device=device)
    assert tam(inputs).shape == torch.Size([batch_size, seq_len, embed_dim])

    # Case 2: Multiple backbones
    num_backbones = 10
    tam = TAM(
        num_backbones,
        True,
        max_rel_pos,
        embed_dim
    )

    tam = tam.to(device)
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


def test_model() -> None:
    batch_size = 64
    device = "cuda"

    hidden_dims = [1024, 512, 512, 1024]
    embed_dim = 1024
    seq_len = 32
    max_relative_position = 10

    config = {
        "MODEL_TEACHER_ARGS": {
            # Input dim of in_proj MLP is predicated upon corresponding feature extractor
            "in_proj_args":
                [
                    # 1st feature extractor
                    {
                        "input_dim": hidden_dims[0],
                        "hidden_dim": [512, 512],
                        "output_dim": embed_dim,
                        "bias": True,
                        "dropout": .5,
                        "hidden_activation": "LeakyReLU",
                        "hidden_activation_args": None,
                        "out_activation": None,
                        "out_activation_args": None,
                        "layer_order": "fc->drop->act",
                    },
                    # 2nd feature extractor
                    {
                        "input_dim": hidden_dims[1],
                        "hidden_dim": [512, 512],
                        "output_dim": embed_dim,
                        "bias": True,
                        "dropout": .5,
                        "hidden_activation": "LeakyReLU",
                        "hidden_activation_args": None,
                        "out_activation": None,
                        "out_activation_args": None,
                        "layer_order": "fc->drop->act",
                    },
                    {
                        "input_dim": hidden_dims[2],
                        "hidden_dim": [512, 512],
                        "output_dim": embed_dim,
                        "bias": True,
                        "dropout": .5,
                        "hidden_activation": "LeakyReLU",
                        "hidden_activation_args": None,
                        "out_activation": None,
                        "out_activation_args": None,
                        "layer_order": "fc->drop->act",
                    },
                    {
                        "input_dim": hidden_dims[3],
                        "hidden_dim": [512, 512],
                        "output_dim": embed_dim,
                        "bias": True,
                        "dropout": .5,
                        "hidden_activation": "LeakyReLU",
                        "hidden_activation_args": None,
                        "out_activation": None,
                        "out_activation_args": None,
                        "layer_order": "fc->drop->act",
                    }
                ],
            "out_proj_args": {
                "input_dim": embed_dim,
                "hidden_dim": [512, 32],
                "output_dim": 1,
                "bias": True,
                "dropout": .5,
                "hidden_activation": "LeakyReLU",
                "hidden_activation_args": None,
                "out_activation": "Sigmoid",
                "out_activation_args": None,
                "layer_order": "fc->drop->act",
            },

            "TAM_args": {
                # "num_blocks": 1,
                "num_heads": 2,
                "relative_attention": True,
                "max_relative_position": max_relative_position,
            }
        },

        "MODEL_STUDENT_ARGS": {

        },

    }

    extracted_features = [
        torch.randn((batch_size, seq_len, hidden_dim), device=device)
        for hidden_dim in hidden_dims
    ]

    with torch.amp.autocast(device, torch.float16):
        model = TAModel(**config["MODEL_TEACHER_ARGS"]).to(device)

        ModelArchInspector(
            model, None, {"x": extracted_features},
            depth=3,
            device="cuda",
            verbose=1,
            mode="train"
        )()

        outs = model(extracted_features)
        print("Output:", outs.shape)


def main() -> None:
    # test_3D_extract_feature()
    test_2D_extract_feature()
    # test_mlp()
    # test_TAM()
    # test_pseudo_label_refiner()
    # test_model()
    return None


if __name__ == '__main__':
    main()
