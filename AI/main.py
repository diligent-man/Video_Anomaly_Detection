import torch

def main() -> None:
    print(torch.cuda.is_available())
    return None

if __name__ == '__main__':
    main()