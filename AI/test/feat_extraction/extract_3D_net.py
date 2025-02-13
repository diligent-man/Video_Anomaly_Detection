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

    video: torch.Tensor = torch.rand((96, 3, 32, 224, 224), device=device, dtype=torch.float16)
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