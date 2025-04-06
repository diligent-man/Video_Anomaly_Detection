from typing import Dict, Any

from torch import Tensor
from torch.nn import Module

from ..nn import MLP
from ...opensrc.pytorch import avail_act
from .HeadOutput import SimpleClassifierOutput


__all__ = ["SimpleClassifier"]


class SimpleClassifier(Module):

    def __init__(self, return_logits: bool = False, *args, **kwargs) -> None:
        super(SimpleClassifier, self).__init__()

        out_act: None | str = kwargs.pop("out_act", None)
        out_act_args: Dict[str, Any] = kwargs.pop("out_act_args", {})

        if out_act is not None:
            assert out_act in avail_act.keys(), ValueError("Specified output activation function is unavailable")
            self.out_act: Module = avail_act[out_act](**out_act_args)
        else:
            self.out_act: None = out_act

        self.classifier: Module = MLP(*args, **kwargs)
        self.return_logits: bool = return_logits

    def forward(self, x: Tensor) -> SimpleClassifierOutput:
        """
        :param x: input shape (*, H_in), where H_in is a number of in_feats
        :return: output shape (*, H_out), where H_out is a number of out_feats
        """
        logits: Tensor = self.classifier(x)
        logits = logits.squeeze(-1)

        if self.out_act is not None:
            preds: Tensor = self.out_act(logits)
        else:
            preds: None = None
        return SimpleClassifierOutput(
            preds=preds,
            logits=logits if self.return_logits else None,
        )
