import typing

import numpy as np
# import matplotlib.pyplot as plt

from . import config

__all__ = [
    "generate_complex_sinusoid",
    "generate_time_domain_impulse"
]


def generate_complex_sinusoid(n: int,
                              freqs: typing.List[float],
                              phases: typing.List[float],
                              bin_offset: float = 0.0):
    """
    Generate a complex sinusoid of length n.
    The sinusoid will be comprised of len(freq) frequencies. Each composite
    sinusoid will have a corresponding phase shift from phases
    """
    t = np.arange(n)
    sig = np.zeros(n, dtype=config.complex_dtype)
    for i in range(len(freqs)):
        sig += np.exp(1j*(2*np.pi*(freqs[i] + bin_offset)/n*t + phases[i]))
    return sig


def generate_time_domain_impulse(n, offsets, widths):

    sig = np.zeros(n, dtype=config.complex_dtype)
    for i in range(len(offsets)):
        offset = offsets[i]
        width = widths[i]
        sig[offset: offset+width] = 1.0
    return sig


if __name__ == "__main__":
    pass
    # fig, axes = plt.subplots(5, 2)
    # print(axes.shape)
    # # fig.tight_layout()
    # for i in range(1, axes.shape[0]+1):
    #     sig1 = generate_complex_sinusoid(1024, [4], [2*np.pi/i])
    #     sig2 = generate_complex_sinusoid(1024, [4], [2*np.pi/i], bin_offset=0.0)
    #     print(np.allclose(sig1, sig2))
    #     axes[i-1, 0].plot(sig1.real)
    #     axes[i-1, 0].plot(sig1.imag)
    #     axes[i-1, 1].plot(sig2.real)
    #     axes[i-1, 1].plot(sig2.imag)
    #     axes[i-1, 0].grid(True)
    #     axes[i-1, 1].grid(True)
    # plt.show()
