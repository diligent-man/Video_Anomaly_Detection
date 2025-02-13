"""
Test dataset and dataloader.
"""
from AI.src.data.dataset import datasets
from AI.src.data.dataloader import dataloaders


def main() -> None:
    dataset_name = "VideoDataset"
    dataloader_name = "DefaultDataLoader"
    assert dataset_name in datasets.keys(), ValueError("Module is not supported")
    assert dataloader_name in dataloaders.keys(), ValueError("Module is not supported")

    dataset = datasets[dataset_name](
            "/home/trong/Downloads/Dataset/VAD/UCF-Crime/Anomaly_videos/Abuse",
            "mp4",
            device="cpu",
            loader="v2"
    )

    dataloader = dataloaders[dataloader_name](
        dataset,
        batch_size=8,
        num_workers=6,
        shuffle=False,
        drop_last=False,
        prefetch_factor=1,
        multiprocessing_context="fork",
    )

    print(dataset)
    print(f"Dataloader len: {len(dataloader)}")

    for i, batch in enumerate(dataloader):
        video_paths, videos = batch
        print(video_paths, videos.shape)
        break
    return None


if __name__ == '__main__':
    main()
