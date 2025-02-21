__all__ = ["EarlyStopping"]


class EarlyStopping(object):
    """
    Currently track val loss as a criterion for early stop.
    """
    def __init__(self,
                 min_val_loss: float = float("inf"),
                 patience: int = 1,
                 min_delta: int = 0
                 ) -> None:
        super(EarlyStopping, self).__init__()
        self.__counter: int = 0
        self.__min_delta: int = min_delta
        self.__patience: int = patience  # Num of epoch that val loss is allowed to increase
        self.__min_val_loss: float = min_val_loss

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.__min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0

        elif val_loss > (self.min_val_loss + self.__min_delta):
            self.counter += 1
            if self.counter >= self.__patience:
                return True
        return False
