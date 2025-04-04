import torch
def main() -> None:
    a = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=torch.float32)

    print(torch.mean(a, dim=1))


    return None

if __name__ == '__main__':
    main()