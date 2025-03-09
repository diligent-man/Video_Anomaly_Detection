from ...runner import Trainer


__all__ = ["BaseCallback"]


class BaseCallback(object):
    """
    Orthodox training model workflow and corresponding callback fn:
                        Init Trainer                                      ->        on_init_end
                            ↓
                       Begin train                                        ->        on_train_begin
                            ↓
                     Run train epoch                                      ->        on_train_epoch_begin
                            ↓
                     Step dataloader                                      ->        on_step_begin
                            ↓
            Accumulate grad with other substeps                           ->        on_substep_end
                            ↓
                    Complete 1 batch                                      ->        on_step_end
                            ↓
                Finish clipping grad                                      ->        on_pre_optimizer_step
                            ↓
                Step optimizer/ scheduler                                 ->        on_optimizer_step
                            ↓
  End train phase by computing metrics (if feasible), train loss          ->        on_train_epoch_end
                            ↓
                        Begin eval                                        ->        on_val_epoch_begin
                            ↓
                      Step dataloader                                     ->        on_step_begin
                            ↓
                    Complete 1 batch                                      ->        on_step_end
                            ↓
End val phase by computing metrics (if feasible), val loss                ->        on_val_epoch_end
                            ↓
End training in case of being trained with specified epochs/ steps        ->        on_train_end
    """
    def on_init_end(self, instance: Trainer) -> None:
        pass

    def on_train_begin(self, instance: Trainer) -> None:
        pass

    def on_train_epoch_begin(self, instance: Trainer) -> None:
        pass

    def on_step_begin(self, instance: Trainer) -> None:
        """Currently, use for both train/ val """
        pass

    def on_substep_end(self, instance: Trainer) -> None:
        pass

    def on_step_end(self, instance: Trainer) -> None:
        """Currently, use for both train/ val """
        pass

    def on_pre_optimizer_step(self, instance: Trainer) -> None:
        pass

    def on_optimizer_step(self, instance: Trainer) -> None:
        pass

    def on_train_epoch_end(self, instance: Trainer) -> None:
        pass

    def on_val_epoch_begin(self, instance: Trainer) -> None:
        pass

    def on_val_epoch_end(self, instance: Trainer) -> None:
        pass

    def on_train_end(self, instance: Trainer) -> None:
        pass

    # def on_save(self, instance: Trainer) -> None:
    #     pass
    #
    # def on_log(self, instance: Trainer) -> None:
    #     pass
