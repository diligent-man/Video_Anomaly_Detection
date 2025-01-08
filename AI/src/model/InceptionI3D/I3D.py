from typing import Any
from collections import OrderedDict

import torch

from .Unit3D import Unit3D
from .MaxPool3dSamePadding import MaxPool3dSamePadding
from .InceptionBlock import InceptionBlock


__all__ = ["inception_i3d"]


class InceptionI3d(torch.nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        https://arxiv.org/pdf/1705.07750v1.pdf.

    Inception architecture, introduced in:
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """
    __VALID_ENDPOINTS = (
        "conv3d_1a_7x7",
        "maxPool3d_2a_3x3",
        "conv3d_2b_1x1",
        "conv3d_2c_3x3",
        "maxPool3d_3a_3x3",
        "mixed_3b",
        "mixed_3c",
        "maxPool3d_4a_3x3",
        "mixed_4b",
        "mixed_4c",
        "mixed_4d",
        "mixed_4e",
        "mixed_4f",
        "maxPool3d_5a_2x2",
        "mixed_5b",
        "mixed_5c",
        "logits",
        "predictions",
    )

    __DEFAULT_INCEPTION_CONFIG = OrderedDict({
        "maxPool3d_2a_3x3": MaxPool3dSamePadding((1, 3, 3), (1, 2, 2)),
        "conv3d_2b_1x1": Unit3D(64, 64, (1, 1, 1)),
        "conv3d_2c_3x3": Unit3D(64, 192, (3, 3, 3), padding=1),

        "maxPool3d_3a_3x3": MaxPool3dSamePadding((1, 3, 3), (1, 2, 2)),
        "mixed_3b": InceptionBlock(192, [64, 96, 128, 16, 32, 32]),
        "mixed_3c": InceptionBlock(sum([64, 128, 32, 32]), [128, 128, 192, 32, 96, 64]),

        "maxPool3d_4a_3x3": MaxPool3dSamePadding((3, 3, 3), (2, 2, 2)),
        "mixed_4b": InceptionBlock(sum([128, 192, 96, 64]), [192, 96, 208, 16, 48, 64]),
        "mixed_4c": InceptionBlock(sum([192, 208, 48, 64]), [160, 112, 224, 24, 64, 64]),
        "mixed_4d": InceptionBlock(sum([160, 224, 64, 64]), [128, 128, 256, 24, 64, 64]),
        "mixed_4e": InceptionBlock(sum([128, 256, 64, 64]), [112, 144, 288, 32, 64, 64]),
        "mixed_4f": InceptionBlock(sum([112, 288, 64, 64]), [256, 160, 320, 32, 128, 128]),

        "maxPool3d_5a_2x2": MaxPool3dSamePadding((2, 2, 2), (2, 2, 2)),
        "mixed_5b": InceptionBlock(sum([256, 320, 128, 128]), [256, 160, 320, 32, 128, 128]),
        "mixed_5c": InceptionBlock(sum([256, 320, 128, 128]), [384, 192, 384, 48, 128, 128])
    })

    def __init__(self,
                 num_classes=400,
                 spatial_squeeze=True,
                 final_endpoint="logits",
                 in_channels=3,
                 dropout_keep_prob=0.5
                 ):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """
        assert final_endpoint in self.__VALID_ENDPOINTS, ValueError(f"Unknown final endpoint {final_endpoint}")
        super(InceptionI3d, self).__init__()
        self.__num_classes = num_classes
        self.__spatial_squeeze = spatial_squeeze
        self.__final_endpoint = final_endpoint
        self.__end_points = self.__build_endpoints(in_channels)

        self.logits = self.__end_points.pop("logits", None)
        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = torch.nn.Dropout(dropout_keep_prob)

    # def replace_logits(self, num_classes: int):
    #     self._num_classes = num_classes
    #     self.logits = Unit3D(384 + 384 + 128 + 128, self._num_classes,
    #                          activation_fn=None,
    #                          use_batch_norm=False,
    #                          use_bias=True,
    #                          name="logits"
    #                          )

    # def extract_features(self, x):
    #     for end_point in self.VALID_ENDPOINTS:
    #         if end_point in self.end_points:
    #             x = self._modules[end_point](x)
    #     return self.avg_pool(x)

    def __build_endpoints(self, in_channels: int) -> dict:
        endpoints = OrderedDict({})

        for endpoint in self.__VALID_ENDPOINTS:
            if endpoint == "conv3d_1a_7x7":
                # First layer
                endpoints[endpoint] = Unit3D(in_channels, 64, (7, 7, 7), (2, 2, 2), (3, 3, 3))
            elif endpoint == "logits":
                # Output layer
                endpoints[endpoint] = Unit3D(
                    384 + 384 + 128 + 128,
                    self.__num_classes,
                    activation_fn=None,
                    use_batch_norm=False,
                    use_bias=True,
                )
            else:
                # Intermediary layers
                endpoints[endpoint] = self.__DEFAULT_INCEPTION_CONFIG[endpoint]

            if endpoint == self.__final_endpoint:
                break

        for endpoint in endpoints.keys():
            if endpoint != "logits":
                self.add_module(endpoint, endpoints[endpoint])
        return endpoints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:
        :return: Logits with shape (B, T, C)
        """
        # use _modules to work with data in parallel
        for end_point in self.__VALID_ENDPOINTS:
            if end_point in self.__end_points:
                x = self._modules[end_point](x)

        x = self.avg_pool(x)
        x = self.dropout(x)

        if self.logits is not None:
            x = self.logits(x)

        if self.__spatial_squeeze:
            x = x.squeeze(3).squeeze(3)
        return x


def inception_i3d(*, weights: str = None, **kwargs: Any) -> InceptionI3d:
    model = InceptionI3d(**kwargs)

    if weights is not None:
        model.load_state_dict(torch.load(weights, weights_only=True))
    return model
