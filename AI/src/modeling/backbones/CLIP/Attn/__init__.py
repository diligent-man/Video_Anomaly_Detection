from .CLIPAttention import CLIPAttention
from .CLIPSdpaAttention import CLIPSdpaAttention
from .CLIPFlashAttention2 import CLIPFlashAttention2

CLIP_ATTENTION_CLASSES = {
    "eager": CLIPAttention,
    "sdpa": CLIPSdpaAttention,
    "flash_attention_2": CLIPFlashAttention2
}

__all__ = ["CLIP_ATTENTION_CLASSES"]
