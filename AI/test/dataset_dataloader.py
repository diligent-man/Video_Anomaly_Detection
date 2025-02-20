"""
Test dataset and dataloader.
"""
from AI.src.data.dataset import datasets
from AI.src.data.dataloader import dataloaders



def main() -> None:
    dataset_name = "VideoFolderDataset"
    dataloader_name = "DefaultDataLoader"
    assert dataset_name in datasets.keys(), ValueError("Module is not supported")
    assert dataloader_name in dataloaders.keys(), ValueError("Module is not supported")

    dataset = datasets[dataset_name](
            "../dataset/ucf-test/unlabed",
            "v3",
            "mp4"
    )

    dataloader = dataloaders[dataloader_name](
        dataset,
        batch_size=32,
        num_workers=6,
        shuffle=True,
        drop_last=False,
        prefetch_factor=1,
        multiprocessing_context="fork",
        collate_fn=collate_fn
    )

    print(dataset)
    print(f"Dataloader len: {len(dataloader)}")

    for i, (videos, labels) in enumerate(dataloader):
        print(videos)
        print(labels)
        print()
        print()
    return None


if __name__ == '__main__':
    main()
