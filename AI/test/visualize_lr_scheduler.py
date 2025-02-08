import torch

from AI.src.utils import visualize_lr
from AI.src.optimizer.scheduler import CustomExponentialLR, CustomSequentialLR, WarmupCosineAnnealingWarmRestarts

def test_update_per_batch(epochs: int,
                          dataloader_len: int,
                          optim: torch.optim.Optimizer
                          ) -> None:
    schedulers = [
        # torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9),

        CustomSequentialLR(optim,
                           [
                               CustomExponentialLR(optim, gamma=1.01),
                               torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, dataloader_len, 1)
                           ],
                           [5 * dataloader_len]
                           ),
        WarmupCosineAnnealingWarmRestarts(optim, dataloader_len, 1, 5 * dataloader_len)
    ]

    for scheduler in schedulers:
        visualize_lr(optim, scheduler, "update_per_batch", epochs, dataloader_len)
    return None


def test_update_per_epoch(epochs: int,
                          dataloader_len: int,
                          optim: torch.optim.Optimizer,
                          ) -> None:
    schedulers = [
        CustomSequentialLR(optim,
                           [
                               CustomExponentialLR(optim, gamma=1.3),
                               torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, dataloader_len, 1)
                           ],
                           [5]
                           ),
        WarmupCosineAnnealingWarmRestarts(optim, dataloader_len, 1, 5)
    ]

    for scheduler in schedulers:
        visualize_lr(optim, scheduler, "update_per_epoch", epochs, dataloader_len)
    return None





def main() -> None:
    epochs = 20
    dataloader_len = 5
    paras = torch.nn.Parameter(torch.rand(3, 224, 224))
    optim = torch.optim.Adam([paras], lr=1e-1)

    test_update_per_batch(epochs, dataloader_len, optim)
    test_update_per_epoch(epochs, dataloader_len, optim)
    return None


if __name__ == '__main__':
    main()