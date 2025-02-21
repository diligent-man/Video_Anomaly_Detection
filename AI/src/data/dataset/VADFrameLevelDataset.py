from torch.utils.data import Dataset


__all__ = ["VADFrameLevelDataset"]


class VADFrameLevelDataset(Dataset):
    raise NotImplementedError

def main() -> None:
    ds = VADFrameLevelDataset(
        normal_root="/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/dataset/ucf-test/unlabeled/normal",
        anomaly_root="/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/dataset/ucf-test/unlabeled/anomaly",
        loader="v4"
    )
    dl = VADVideoLevelDataLoader(ds, 3, True, 1, num_workers=4)
    print(ds)
    inps, labels = next(iter(dl))
    print(inps.shape, labels.shape)
    return None


if __name__ == '__main__':
    main()