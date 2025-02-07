import torch
from AI.src.utils import load_video_v2
from AI.src.data.dataset import datasets


def main() -> None:
    dataset_name = "VideoDataset"
    assert dataset_name in datasets, ValueError("Module is not supported")

    dataset = datasets[dataset_name](
            "/home/trong/Downloads/Dataset/VAD/UCF-Crime/Anomaly_videos/Abuse",
            "mp4",
            device="cpu",
            loader=load_video_v2
    )

    dataloader = torch.utils.data.DataLoader(
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
