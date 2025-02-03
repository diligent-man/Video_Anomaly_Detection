import warnings
import itertools

from typing import Any, Dict, Iterable, List


__all__ = ["DotDict"]


class DotDict(dict):
    """
    Recursively converts nested dictionaries to dotted dictionary.
    """
    _key_error_handling: str
    _depth: int = 0

    def __init__(self,
                 in_dict: Dict[str, Any],
                 depth: int = 0,
                 **kwargs
                 ):
        """
        :param in_dict: input dictionary
        :param key_error_handling: ["raise" | "warn"]
        """
        if not isinstance(in_dict, dict):
            raise ValueError(f"Incorrect input type: {type(in_dict)}")

        self._key_error_handling: str = kwargs.get("key_error_handling", "raise")
        self._depth: int = depth
        super(DotDict, self).__init__()

        for k, v in in_dict.items():
            k = self._preprocess_key(k)

            if isinstance(v, (list, tuple, set)):
                v = self._remove_duplicated_dicts(v)
                setattr(self, k, [DotDict(x, depth+1, **kwargs) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, DotDict(v, depth+1, **kwargs) if isinstance(v, dict) else v)

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

    @staticmethod
    def _preprocess_key(k: str) -> str:
        k = k.replace("-", "_")
        k = k.replace(" ", "_")
        k = k.lower()
        return k

    @staticmethod
    def _remove_duplicated_dicts(iterable: Iterable) -> List[Any]:
        dict_eles: List[Dict[str, Any]] = [x for x in iterable if isinstance(x, dict)]
        dict_eles = [ele[0] for ele in itertools.groupby(dict_eles)]

        non_dict_eles: List[Any] = [x for x in iterable if not isinstance(x, dict)]

        result: List[Any] = [*dict_eles, *non_dict_eles]
        return result
