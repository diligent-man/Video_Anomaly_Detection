import torch

from AI.src.modeling.necks import TemporalAggregation


def main() -> None:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    B, seq_len, hid_dim, embed_dim, max_rel_pos = 3, 32, 768, 768, 4

    # Case 1: Single backbone, cross-c2c term will be omitted
    num_backbones = [1, 2, 3, 10]
    num_heads = [1, 2, 8, 12]

    for num_backbone in num_backbones:
        for num_head in num_heads:
            temporal_aggregation = TemporalAggregation(
                num_backbone,
                num_head,
                hid_dim,
                True,
                True,
                max_rel_pos
            ).to(device)
            x = torch.rand((num_backbone, B, seq_len, hid_dim), device=device)

            out = temporal_aggregation(x)
            assert out.shape == torch.Size([B, seq_len, embed_dim])
    print("Test complete !!!")
    return None


if __name__ == '__main__':
    main()
