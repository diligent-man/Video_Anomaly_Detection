import os
from multiprocessing.pool import Pool
from typing import List, Tuple


class Multiprocessor:
    """
    This class use Pool() for multiprocessing
    """

    def __init__(self, lower: int, upper: int, fixed_configurations: tuple,
                 processes: int = os.cpu_count(), process_counter: bool = True
                 ):
        assert lower < upper, 'Lower bound must be less than upper bound'

        self.__pool = Pool(processes=processes)
        self.__lower = lower;
        self.__upper = upper
        self.__fixed_configurations = fixed_configurations  # all similar paras needed for each process
        self.__configurations = self.__multiprocess_splitter(processes, process_counter)

    @property
    def configurations(self):
        return self.__configurations

    @configurations.setter
    def configurations(self, value):
        self.__configurations = value

    def __multiprocess_splitter(self, processes: int, process_counter: bool) -> List[Tuple]:
        if processes == 1:
            if process_counter:
                return [(self.__lower, self.__upper, *self.__fixed_configurations, 0)]
            else:
                return [(self.__lower, self.__upper, *self.__fixed_configurations)]

        else:
            # This process is similar to minibatch splitting
            configurations = []
            interval = int(round((self.__upper - self.__lower) / processes))

            # last process for handling residual interval
            for i in range(processes - 1):
                if process_counter:
                    configurations.append((self.__lower + i * interval, self.__lower + (i + 1) * interval,
                                           *self.__fixed_configurations, i))
                else:
                    configurations.append(
                        (self.__lower + i * interval, self.__lower + (i + 1) * interval, *self.__fixed_configurations))

                # Early stopping cond
                if self.__lower + (i + 2) * interval > self.__upper:
                    break

            # Handling the residual interval
            if process_counter:
                configurations.append(
                    (self.__lower + (i + 1) * interval, self.__upper, *self.__fixed_configurations, i + 1))
            else:
                configurations.append((self.__lower + (i + 1) * interval, self.__upper, *self.__fixed_configurations))
            return configurations

    def __call__(self, func):
        self.__pool.starmap(func=func, iterable=self.__configurations)