import os
from torchvision.transforms.v2 import Transform
from torchvision.models import WeightsEnum, Weights

__all__ = ["S3D_Weights"]


class S3D_Weights(WeightsEnum):
    KINETICS400_V1 = Weights(
        url=os.path.join(os.path.dirname(os.getcwd()), "weights", "S3D", "RGB_Kinetics400.pth"),
        # url="https://download.pytorch.org/models/s3d-d76dad2f.pth",
        transforms=Transform(),

        # from
        meta={
            "min_size": (224, 224),
            "min_temporal_size": 14,
            "recommend_inp_transform": {
                "crop_size": (224, 224),
                "resize_size": (256, 256),
            },
            "recipe": "https://github.com/pytorch/vision/tree/main/references/video_classification#s3d",
            "_docs": (
                "The weights aim to approximate the accuracy of the paper. The accuracies are estimated on clip-level "
                "with parameters `frame_rate=15`, `clips_per_video=1`, and `clip_len=128`."
            ),
            "_metrics": {
                "Kinetics-400": {
                    "acc@1": 68.368,
                    "acc@5": 88.050,
                }
            },
            "_file_size": 31.972,
        }
    )
    DEFAULT = KINETICS400_V1
