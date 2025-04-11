from AI.src.data.dataset import VADFrameLevelDataset


def main() -> None:
    ds = VADFrameLevelDataset(
        "/home/trong/Downloads/Dataset/VAD/final/test",
        "label.csv",
        "v4",
    )

    print(ds.__getitem__(0))

    return None


if __name__ == '__main__':
    main()