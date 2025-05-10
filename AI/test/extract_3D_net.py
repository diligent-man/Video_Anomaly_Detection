"""
Test feature extractors defined in NET_3D
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
    MultiBackboneForwarder,
    NET_3D_REDUCE,
    DEFAULT_3D_REDUCE,
    NET_DEFAULT_CONFIG,
)

TEST_DATA = {
    "rgb_i3d": "../weights/I3D/RGB_Kinetics400.pt",
    "s3d": "../weights/S3D/RGB_Kinetics400.pth",
}

TRACER_ARGS = {
    "rgb_i3d": {},
    "s3d": {},
}


def main() -> None:
    trace: bool = True
    inspect_model: bool = True
    compile_model: bool = False

    B, S, C, T = 1, 32, 3, 13
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with (torch.autocast(device_type=device, dtype=torch.float16)):
        for name, weights in TEST_DATA.items():
            model, _, return_node, dummy_input = NET_DEFAULT_CONFIG[name].values()
            model: torch.nn.Module = model(weights)

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
            model_forwarder = MultiBackboneForwarder(model, name, NET_3D_REDUCE[DEFAULT_3D_REDUCE])

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
