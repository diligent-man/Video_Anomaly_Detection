"""
Test feature extractors defined in NET_2D
More feature extractors can be found at: https://github.com/v-iashin/video_features/tree/master
"""
import gc
import torch

from AI.src.utils.Tracer import LeafModuleAwareTracer
from AI.src.utils import (
    ModelArchInspector,
    ANSIColor,
    create_feature_extractor
)
from AI.src.modeling.backbones import (
    ModelForwarder,
    NET_2D_REDUCE,
    DEFAULT_2D_REDUCE,
    NET_DEFAULT_CONFIG,
)

TEST_DATA = {
    "clip_vit/b16": "../weights/CLIP/vit-base-patch16-224",
    "clip_vit/b32": "../weights/CLIP/vit-base-patch32-224",
    "clip_vit/l14": "../weights/CLIP/vit-large-patch14-224",
    "clip_vit/l14_336": "../weights/CLIP/vit-large-patch14-336"
}

TRACER_ARGS = {
    "clip_vit/b16": {"concrete_args": {"return_loss": None, "return_dict": None}},
    "clip_vit/b32": {"concrete_args": {"return_loss": None, "return_dict": None}},
    "clip_vit/l14": {"concrete_args": {"return_loss": None, "return_dict": None}},
    "clip_vit/l14_336": {"concrete_args": {"return_loss": None, "return_dict": None}}
}


def main() -> None:
    trace: bool = True
    inspect_model: bool = False
    compile_model: bool = False

    B, S, C, T = 1, 32, 3, 15
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with (torch.autocast(device_type=device, dtype=torch.float16)):
        for name, weights in TEST_DATA.items():
            model, _, return_node, dummy_input = NET_DEFAULT_CONFIG[name].values()
            model: torch.nn.Module = model(weights)
            torch.save(model, r"C:\Users\NDT\Downloads\model.pt")

            if inspect_model:
                ModelArchInspector(
                    model, dummy_input,
                    depth=5,
                    device="cuda",
                    verbose=1,
                    mode="eval",
                )()

            if trace:
                tracer: LeafModuleAwareTracer = LeafModuleAwareTracer()
                graph: torch.fx.graph.Graph = tracer.trace(model, **TRACER_ARGS[name])

                print(f"{ANSIColor().RED}Trace result{ANSIColor().RESET}")
                graph.print_tabular()
                print()
                print()

            model: torch.fx.GraphModule = create_feature_extractor(model, return_node, **TRACER_ARGS[name])

            for para in reversed(list(model.parameters())):
                para.requires_grad = False

            print(f"{ANSIColor().RED}Feature layers{ANSIColor().RESET}\n")
            model.graph.print_tabular()
            print()

            if compile_model:
                model.compile()

            x: torch.Tensor = torch.rand((B, S, C, T, *dummy_input[-2:]), device=device, dtype=torch.float16)
            model_forwarder = ModelForwarder(model, name, NET_2D_REDUCE[DEFAULT_2D_REDUCE])

            features = model_forwarder(x)

            print(f"""{ANSIColor().CYAN}Test feature extractor{ANSIColor().RESET}
    Feature extractor: {model.__class__.__name__}
    Input: {x.shape}
    Output: {features.shape}
######################################################################################################################""")
            gc.collect()
            torch.cuda.empty_cache()
    return None


if __name__ == '__main__':
    main()
