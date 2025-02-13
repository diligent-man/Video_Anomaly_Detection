import torch

from AI.src.modeling.nn import MLP


def main() -> None:
    seq_len: int = 32
    embed_dim: int = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mlp: MLP = MLP(embed_dim,512,[512, 512],
        "LeakyReLU",None,
        None,None,
        True,.5,"fc->act->drop").to(device)
    features = torch.rand((1, seq_len, embed_dim), device=device)

    with torch.autocast(device, torch.float16):
        outputs: torch.Tensor = mlp(features)

    print(f"""Test MLP result:
    Input shape: {features.shape}
    Output shape: {outputs.shape}""")
    return None


if __name__ == '__main__':
    main()