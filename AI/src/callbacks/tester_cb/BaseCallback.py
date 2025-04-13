from AI.src.runner import Tester


__all__ = ["BaseCallback"]


class BaseCallback(object):
    """
    Orthodox training model workflow and corresponding callback fn:
                        Init Tester            ->        on_init_end
                            ↓
                        Begin test             ->        on_begin
                            ↓
                     Step dataloader           ->        on_step_begin
                            ↓
                     Complete batch           ->         on_step_end
                            ↓
                       End testing            ->         on_end
    """
    def on_init_end(self, instance: Tester) -> None:
        pass

    def on_begin(self, instance: Tester) -> None:
        pass

    def on_step_begin(self, instance: Tester) -> None:
        """Currently, use for both train/ val """
        pass

    def on_step_end(self, instance: Tester) -> None:
        """Currently, use for both train/ val """
        pass

    def on_end(self, instance: Tester) -> None:
        pass
