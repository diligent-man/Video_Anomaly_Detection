import copy
import warnings
import itertools
import overrides

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
                 ) -> None:
        """
        :param in_dict: input dictionary
        :param key_error_handling: ["raise" | "warn"]
        """
        if not isinstance(in_dict, dict):
            raise ValueError(f"Incorrect input type: {type(in_dict)}")

        self._depth: int = depth
        self._key_error_handling: str = kwargs.get("key_error_handling", "raise")
        super(DotDict, self).__init__()

        for k, v in in_dict.items():
            k: str = self._preprocess_key(k, depth, kwargs.get("capitalize_first_level_key", True if depth == 0 else False))

            if isinstance(v, (list, tuple, set)):
                v = self._remove_duplicated_dicts(v)
                setattr(self, k, [DotDict(x, depth + 1, **kwargs) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, DotDict(v, depth + 1, **kwargs) if isinstance(v, dict) else v)

    def __getattr__(self, k: str) -> Any:
        try:
            return self[self._preprocess_key(k, self._depth-1)]
        except KeyError as e:
            if self._key_error_handling == "raise":
                raise e
            elif self._key_error_handling == "warn":
                warnings.warn(f"{self.__class__.__name__} has no attribute '{k}' so returns None")

    def __dict__(self) -> Dict[str, Any]:
        return self._parse_dict(dict(self))

    @overrides.override
    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    @overrides.override
    def __delattr__(self, k: str) -> None:
        try:
            del self[k]
        except KeyError:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '{k}'")

    def __deepcopy__(self, memo=None) -> object:
        new_obj: DotDict = type(self)(self)  # Other syntax: self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj

        for k, v in self.items():
            setattr(new_obj, k, copy.deepcopy(v, memo))
        return new_obj

    @staticmethod
    def _preprocess_key(k: str, depth: int, capitalize_first_level_key: bool = False) -> str:
        k = k.replace("-", "_")
        k = k.replace(" ", "_")
        k = k.capitalize() if capitalize_first_level_key or depth == 0 else k
        return k

    @staticmethod
    def _remove_duplicated_dicts(iterable: Iterable) -> List[Any]:
        dict_eles: List[Dict[str, Any]] = [x for x in iterable if isinstance(x, dict)]
        dict_eles = [ele[0] for ele in itertools.groupby(dict_eles)]

        non_dict_eles: List[Any] = [x for x in iterable if not isinstance(x, dict)]

        result: List[Any] = [*dict_eles, *non_dict_eles]
        return result

    @staticmethod
    def _is_public(attr_name: str) -> bool:
        if not (attr_name.startswith("_") or attr_name.startswith("__")):
            return True
        else:
            return False

    # TODO: being duplicated with get_dict. Fix later on
    def _parse_dict(self, in_dict: Dict[str, Any]) -> Dict[str, Any]:
        parsed_dict: Dict[str, Any] = {}

        for k, v in in_dict.items():
            if self._is_public(k):
                if isinstance(v, dict):
                    v: Dict[str, Any] = self._parse_dict(v)
                elif isinstance(v, (tuple, list)):
                    v: List[Any] = [self._parse_dict(x) if isinstance(x, dict) else x for x in v]

                parsed_dict[k] = v
        return parsed_dict

    def get_dict(self, k: str = None) -> Dict[str, Any]:
        return_dict: Dict[str, Any] = {}

        if k is not None and self.get(k) is not None:
            for k, v in self[k].items():
                if self._is_public(k):
                    if isinstance(v, dict):
                        v: Dict[str, Any] = self._parse_dict(v)
                    elif isinstance(v, (tuple, list)):
                        v: List[Any] = [self._parse_dict(x) if isinstance(x, dict) else x for x in v]

                    return_dict[k] = v
        else:
            return_dict = self.__dict__()
        return return_dict
