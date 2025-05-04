from typing import Dict, List, Any
from collections import defaultdict

import math

__all__ = ["VideoCache"]


class VideoCache(object):
    """
    Simple cache for test VAD model with .pt encoded video
    """
    __cache: Dict[int, Dict[str, List[Any]]]

    __cached_k: List[str] = None
    __batch_worker: List[int] = [16, 16, 16, 16,
                                 14, 14,
                                 12, 12, 12
                                 ]
    __batch_thres: List[int | float] = [0, 1000, 2000, 3000, 4500,
                                        6000, 10000,
                                        15000, 20000, math.inf
                                        ]

    def __init__(self,
                 batch_thres: List[int] = None,
                 batch_worker: List[int] = None,
                 ) -> None:
        if batch_thres is not None and batch_worker is not None:
            assert len(batch_worker) == len(batch_thres), ValueError("num worker should equal num threshold")
            batch_thres = [0] + batch_thres
            self.__batch_thres = self._convert_inf_str(batch_thres)
            self.__batch_worker = batch_worker

        super(VideoCache, self).__init__()
        self.__cache = {k: defaultdict(list) for k in self.__batch_thres[1:]}

    def __repr__(self) -> str:
        return f"""Batch thres: {self.__batch_thres}
Batch worker: {self.__batch_worker}"""

    @property
    def batch_thres(self) -> List[int | float]:
        return self.__batch_thres

    @property
    def batch_worker(self) -> List[int]:
        return self.__batch_worker

    @staticmethod
    def _convert_inf_str(batch_thres: List[int | str]) -> List[int | float]:
        return list(map(lambda x: math.inf if x == "inf" else x, batch_thres))

    def cache(self, video_len: int, k_lst: List[str], v_lst: List[Any]) -> None:
        if self.__cached_k is None:
            self.__cached_k = k_lst

        for i in range(len(self.__batch_thres)-1):
            if self.__batch_thres[i] <= video_len <= self.__batch_thres[i+1]:
                for k, v in zip(k_lst, v_lst):
                    self.__cache[self.__batch_thres[i+1]][k].append(v)

    def get_cache(self) -> Dict[str, Any] | None:
        result: Dict[str, Any] | None = None

        for i in range(1, len(self.__batch_thres)):
            first_k = self.__cached_k[0]

            if len(self.__cache[self.__batch_thres[i]][first_k]) == self.__batch_worker[i-1]:
                result = self.__cache[self.__batch_thres[i]]
                result["batch_worker"] = self.__batch_worker[i-1]
                self.__cache[self.__batch_thres[i]] = defaultdict(list)
                break
        return result

    def get_remains(self) -> Dict[str, Any]:
        for i in range(1, len(self.__batch_thres)):
            first_k = self.__cached_k[0]

            if len(self.__cache[self.__batch_thres[i]][first_k]) > 0:
                result: Dict[str, Any] = self.__cache[self.__batch_thres[i]]
                result["batch_worker"] = min(self.__batch_worker[i-1], len(self.__cache[self.__batch_thres[i]][first_k]))
                yield result
