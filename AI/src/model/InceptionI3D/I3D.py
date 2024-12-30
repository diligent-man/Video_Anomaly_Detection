import torch

from .Unit3D import Unit3D
from .MaxPool3dSamePadding import MaxPool3dSamePadding
from .InceptionBlock import InceptionBlock


__all__ = ["InceptionI3d"]


class InceptionI3d(torch.nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        https://arxiv.org/pdf/1705.07750v1.pdf.

    Inception architecture, introduced in:
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self,
                 num_classes=400,
                 spatial_squeeze=True,
                 final_endpoint="Logits",
                 name="Inception_I3D",
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

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels, 64, (7, 7, 7), (2, 2, 2), (3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding((1, 3, 3), (1, 2, 2))

        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(64, 64, (1, 1, 1), name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(64, 192, (3, 3, 3), padding=1, name=name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding((1, 3, 3), (1, 2, 2))
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionBlock(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionBlock(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding((3, 3, 3), (2, 2, 2))
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionBlock(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionBlock(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionBlock(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionBlock(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionBlock(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding((2, 2, 2), (2, 2, 2))
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionBlock(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionBlock(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], name + end_point)
        if self._final_endpoint == end_point:
            return

        self.avg_pool = torch.nn.AvgPool3d((2, 7, 7), (1, 1, 1))

        self.dropout = torch.nn.Dropout(dropout_keep_prob)

        self.logits = Unit3D(384 + 384 + 128 + 128, self._num_classes,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name=name + self._final_endpoint
                             )
        self._build()

    def _build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def replace_logits(self, num_classes: int):
        self._num_classes = num_classes
        self.logits = Unit3D(384 + 384 + 128 + 128, self._num_classes,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name="logits"
                             )

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x:
        :return: Logits with shape (B, T, C)
        """
        # use _modules to work with data in parallel
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)

        x = self.logits(self.dropout(self.avg_pool(x)))

        if self._spatial_squeeze:
            x = x.squeeze()
        return x
