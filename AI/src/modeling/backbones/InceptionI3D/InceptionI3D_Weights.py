import os
from torchvision.transforms.v2 import Transform
from torchvision.models import WeightsEnum, Weights

__all__ = ["InceptionI3D_Weights"]


class InceptionI3D_Weights(WeightsEnum):
    KINETICS400_V1 = Weights(
        url=os.path.join(os.path.dirname(os.getcwd()), "weights", "I3D", "RGB_Kinetics400.pt"),
        transforms=Transform(),
        meta={
            "min_size": (224, 224),
            "min_temporal_size": 14,
            "recommend_inp_transform": {
                "crop_size": (224, 224),
                "resize_size": (256, 256),
                "rescale": (-1, 1),
            },
            "_docs": (),
            "_metrics": {
                "Kinetics-400": {
                    "acc@1": 71.1,
                    "acc@5": 89.3,
                }
            }
        }
    )
    DEFAULT = KINETICS400_V1
