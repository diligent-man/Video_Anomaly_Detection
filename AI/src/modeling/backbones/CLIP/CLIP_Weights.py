import os
from torchvision.transforms.v2 import Transform
from torchvision.models import WeightsEnum, Weights

__all__ = ["CLIP_Weights"]


class CLIP_Weights(WeightsEnum):
     BASE_PATCH16_224 = Weights(
        url=os.path.join("weights", "CLIP", "vit-base-patch16-224"),
        transforms=Transform(),
        meta={
            "min_size": (224, 224),
            "min_batch": 1,
            "recommend_inp_transform": {"resize_size": (256, 256)},
            "_dataset": "YFCC100M",
            "_docs": (),
            "_metrics": {
                "Note": """Linear probe avg over 27 datasets as described in Learning Transferable
                           Visual Models From Natural Language Supervision 12.
                           ViT versions was initially pretrained on ImageNet21k
                        """,

                "vit-base-patch16": {"avg_clip_score": 80.92},
                "vit-base-patch32": {"avg_clip_score": 78.05},
                "vit-large-patch14": {"avg_clip_score": 84.21},
                "vit-large-patch14-336px": {"avg_clip_score": 85.05}
            }
        }
    )

     BASE_PATCH32_224 = Weights(
         url=os.path.join("weights", "CLIP", "vit-base-patch32-224"),
         transforms=Transform(),
         meta={
             "min_size": (224, 224),
             "min_batch": 1,
             "recommend_inp_transform": {"resize_size": (256, 256)},
             "dataset": "YFCC100M",
             "_docs": (),
             "_metrics": {
                 "Note": """Linear probe avg over 27 datasets as described in Learning Transferable
                                Visual Models From Natural Language Supervision 12.
                                ViT versions was initially pretrained on ImageNet21k
                             """,

                 "vit-base-patch16": {"avg_clip_score": 80.92},
                 "vit-base-patch32": {"avg_clip_score": 78.05},
                 "vit-large-patch14": {"avg_clip_score": 84.21},
                 "vit-large-patch14-336px": {"avg_clip_score": 85.05}
             }
         }
     )

     LARGE_PATCH14_224 = Weights(
         url=os.path.join("weights", "CLIP", "vit-large-patch14-224"),
         transforms=Transform(),
         meta={
             "min_size": (224, 224),
             "min_batch": 1,
             "recommend_inp_transform": {"resize_size": (224, 224),},
             "dataset": "YFCC100M",
             "_docs": (),
             "_metrics": {
                 "Note": """Linear probe avg over 27 datasets as described in Learning Transferable
                                Visual Models From Natural Language Supervision 12.
                                ViT versions was initially pretrained on ImageNet21k
                             """,

                 "vit-base-patch16": {"avg_clip_score": 80.92},
                 "vit-base-patch32": {"avg_clip_score": 78.05},
                 "vit-large-patch14": {"avg_clip_score": 84.21},
                 "vit-large-patch14-336px": {"avg_clip_score": 85.05}
             }
         }
     )

     LARGE_PATCH14_336 = Weights(
         url=os.path.join("weights", "CLIP", "vit-base-patch14-336"),
         transforms=Transform(),
         meta={
             "min_size": (336, 336),
             "min_batch": 1,
             "recommend_inp_transform": {"resize_size": (336, 336)},
             "dataset": "YFCC100M",
             "_docs": (),
             "_metrics": {
                 "Note": """Linear probe avg over 27 datasets as described in Learning Transferable
                                Visual Models From Natural Language Supervision 12.
                                ViT versions was initially pretrained on ImageNet21k
                             """,

                 "vit-base-patch16": {"avg_clip_score": 80.92},
                 "vit-base-patch32": {"avg_clip_score": 78.05},
                 "vit-large-patch14": {"avg_clip_score": 84.21},
                 "vit-large-patch14-336px": {"avg_clip_score": 85.05}
             }
         }
     )
     DEFAULT = BASE_PATCH16_224
