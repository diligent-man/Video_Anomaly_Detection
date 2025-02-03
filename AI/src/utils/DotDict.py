import warnings

from typing import Any, Dict


__all__ = ["DotDict"]


class DotDict(dict):
    """
    Recursively converts nested dictionaries to dotted dictionary.
    """
    __key_error_handling: str

    def __init__(self,
                 in_dict: Dict[str, Any],
                 **kwargs
                 ):
        """
        :param in_dict: input dictionary
        :param key_error_handling: ["raise" | "warn"]
        TODO: Check duplicated objs in sequence
        """
        if not isinstance(in_dict, dict):
            raise ValueError(f"Incorrect input type: {type(in_dict)}")

        self.__key_error_handling = kwargs.get("key_error_handling", "raise")
        super(DotDict, self).__init__()

        for k, v in in_dict.items():
            k = self._preprocess_key(k)

            if isinstance(v, (list, tuple, set)):
                # set an attribute for every item in the Sequence
                setattr(self, k, [DotDict(x, **kwargs) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, DotDict(v, **kwargs) if isinstance(v, dict) else v)

    def __getattr__(self, k: str) -> Any:
        try:
            return self[k]
        except KeyError as e:
            if self.__key_error_handling == "raise":
                raise e
            elif self.__key_error_handling == "warn":
                warnings.warn(f"{self.__class__.__name__} has no attribute '{k}' so returns None")

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def __delattr__(self, k: str) -> None:
        try:
            del self[k]
        except KeyError:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '{k}'")

    @property
    def depth(self):
        return self.__depth

    @staticmethod
    def _preprocess_key(k: str) -> str:
        k = k.replace("-", "_")
        k = k.replace(" ", "_")
        k = k.lower()
        return k
