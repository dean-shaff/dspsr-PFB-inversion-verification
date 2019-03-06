# util.py
import typing
from contextlib import contextmanager

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from pfb.formats import DADAFile

__all__ = [
    "debug_plot",
    "get_time_shift",
    "get_nseries"
]


def get_nseries(*dada_objs: typing.Tuple[DADAFile],
                fft_size: int = 229376) -> int:
    min_size = min(*[obj.ndat for obj in dada_objs])
    nseries = int(min_size // fft_size)
    return nseries, min_size


def get_time_shift(a: np.ndarray,
                   b: np.ndarray) -> typing.Tuple[np.ndarray, int]:
    a = a.copy()
    b = b.copy()
    a /= np.amax(a)
    b /= np.amax(b)
    xcorr = scipy.signal.fftconvolve(a, np.conj(b)[::-1], mode="full")
    mid_idx = int(xcorr.shape[0] // 2)
    max_arg = np.argmax(xcorr)
    offset = max_arg - mid_idx
    return a, np.roll(b, abs(offset))


@contextmanager
def debug_plot(arrs, **kwargs):
    """
    Plot some data, waiting for user to prompt
    Plot arr. Array should have dimensions (ndat, nchan, npol)
    """
    plt.ion()
    polar = kwargs.pop("polar", False)
    fig_objs = []
    axes_objs = []
    for arr in arrs:
        iscomplex = np.iscomplexobj(arr)
        ndim = 2 if iscomplex else 1

        ndat, nchan, npol = arr.shape
        fig, axes = plt.subplots(nchan, ndim*npol,
                                 figsize=(3*ndim*npol, 4*nchan + 2))
        fig.tight_layout()
        if axes.ndim == 1:
            axes = [axes]
        fig_objs.append(fig)
        axes_objs.append(axes)

        for ichan in range(nchan):
            for ipol in range(npol):
                if iscomplex:
                    if polar:
                        axes[ichan][ipol*2].plot(
                            np.abs(arr[:, ichan, ipol]), **kwargs)
                        axes[ichan][ipol*2].set_title(
                            f"chan {ichan}, pol {ipol}, magnitude")
                        axes[ichan][ipol*2].grid(True)
                        axes[ichan][ipol*2 + 1].plot(
                            np.angle(arr[:, ichan, ipol]), **kwargs)
                        axes[ichan][ipol*2 + 1].set_title(
                            f"chan {ichan}, pol {ipol}, phase")
                        axes[ichan][ipol*2 + 1].grid(True)
                    else:
                        axes[ichan][ipol*2].plot(
                            arr[:, ichan, ipol].real, **kwargs)
                        axes[ichan][ipol*2].set_title(
                            f"chan {ichan}, pol {ipol}, real")
                        axes[ichan][ipol*2].grid(True)
                        axes[ichan][ipol*2 + 1].plot(
                            arr[:, ichan, ipol].imag, **kwargs)
                        axes[ichan][ipol*2 + 1].set_title(
                            f"chan {ichan}, pol {ipol}, imag")
                        axes[ichan][ipol*2 + 1].grid(True)
                else:
                    axes[ichan][ipol].plot(arr[:, ichan, ipol], **kwargs)
                    axes[ichan][ipol].set_title(f"chan {ichan}, pol {ipol}")
                    axes[ichan][ipol].grid(True)

    yield fig_objs, axes_objs
    plt.ioff()
