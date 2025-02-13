"""
Test PseudoLabelRefiner
"""
import torch

from AI.src.modeling.nn import MLP
from AI.src.modeling.postprocessing import PseudoLabelRefiner


def main() -> None:
    B, seq_len, embed_dim, mask_threshold = 1, 32, 512, .9
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mlp = MLP(
        embed_dim,
        1,
        [512, 32],
        "ReLU",
        None,
        "Sigmoid",
        None,
        True, 0.5
    ).to(device)
    pseudo_label_refiner: PseudoLabelRefiner = PseudoLabelRefiner(3)

    # Output shape from TAN: [batch_size, seq_len, embed_dim]
    x: torch.Tensor = torch.rand((B, seq_len, embed_dim), device=device)

    with torch.amp.autocast(device, torch.float16):
        # [batch_size, seq_len, 1]
        preds: torch.Tensor = mlp(x)
        preds = pseudo_label_refiner(preds)
        mask = torch.where(preds > mask_threshold, 1, 0)

        print(f"""Segment-level prediction/ soft pseudo-labels
{mask.squeeze()}""")
    return None


if __name__ == '__main__':
    main()
