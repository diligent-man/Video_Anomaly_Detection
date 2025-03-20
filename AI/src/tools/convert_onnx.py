import torch
from AI.src.modeling.backbones import NET_DEFAULT_CONFIG


def main() -> None:
    RETURN_NODES = {
        "rgb_i3d": {"squeeze_1": "output"},
        "s3d": {"mean": "output"},

        "clip_vit/b16": {"vision_model": "output"},
    }

    TRACER_ARGS = {
        "rgb_i3d": {},
        "s3d": {},

        "clip_vit/b16": {"concrete_args": {"return_loss": None, "return_dict": None}},
    }

    SAVE_PATH = {
        "rgb_i3d": r"C:\Users\NDT\Downloads\rgb_i3d.pt",
        "s3d": r"C:\Users\NDT\Downloads\s3d.pt",
        "clip_vit/b16": r"C:\Users\NDT\Downloads\clip_vit_b16.pt",
    }

    for (model_name, trace_arg), return_node, save_path in zip(TRACER_ARGS.items(),
                                                               RETURN_NODES.values(),
                                                               SAVE_PATH.values()
                                                               ):
        model, weight, _, dummy_input = NET_DEFAULT_CONFIG[model_name].values()
        model: torch.nn.Module = model(weight)

        torch.onnx.export(
            model,
            torch.rand(dummy_input),
            save_path,
            opset_version=16,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            export_params=True,
            do_constant_folding=True,
        )
    return None


if __name__ == '__main__':
    main()
