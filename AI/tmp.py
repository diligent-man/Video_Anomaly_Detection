import numpy as np

from matplotlib import pyplot as plt

from scipy.signal import find_peaks
from scipy.datasets import electrocardiogram

plt.switch_backend("tkagg")


def main() -> None:
    freq: int = 360  # from doc
    ds: np.ndarray = electrocardiogram()[: 2 * 360]  # get first 2s data
    t = np.arange(ds.shape[0]) / freq

    peaks, props = find_peaks(
        x=ds,
        
    )

    # plt.plot(t, ds)
    # plt.show()
    return None


if __name__ == '__main__':
    main()