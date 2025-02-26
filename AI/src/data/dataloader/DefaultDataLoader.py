from torch.utils.data import DataLoader

__all__ = ["DefaultDataLoader"]


class DefaultDataLoader(DataLoader):
    def __repr__(self) -> str:
        return self.dataset.__repr__()
