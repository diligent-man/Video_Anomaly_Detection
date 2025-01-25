from torch import (
    complex64,
    complex128,

    float16,
    float32,
    float64,

    uint8,

    int8,
    int16,
    int32,
    int64,
)

available_dtype = {
    "complex64": complex64,
    "complex128": complex128,

    "float16": float16,
    "float32": float32,
    "float64": float64,

    "uint8": uint8,

    "int8": int8,
    "int16": int16,
    "int32": int32,
    "int64": int64
}
