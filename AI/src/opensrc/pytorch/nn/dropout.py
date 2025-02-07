from torch.nn import (
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    AlphaDropout,
    FeatureAlphaDropout
)

available_dropout = {
    "Dropout": Dropout,
    "Dropout1d": Dropout1d,
    "Dropout2d": Dropout2d,
    "Dropout3d": Dropout3d,
    "AlphaDropout": AlphaDropout,
    "FeatureAlphaDropout": FeatureAlphaDropout
}
