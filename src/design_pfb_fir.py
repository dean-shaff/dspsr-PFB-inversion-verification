import logging

import numpy as np
import scipy.signal

module_logger = logging.getLogger(__name__)

__all__ = [
    "design_pfb_fir"
]


def design_pfb_fir(ntaps: int,
                   nchan: int,
                   oversampling_factor: str) -> np.ndarray:
    """
    Design an FIR lowpass filter for application in PFB

    Args:
        ntaps: Number of taps. Needs to be odd.
    """
    os_nu, os_de = [int(s) for s in oversampling_factor.split("/")]
    os = os_nu / os_de
    # freq_pass marks the start of the transition band
    freq_pass = 1./nchan
    # freq_stop marks the end of the the transition band
    freq_stop = (1*(2*os - 1))/nchan
    # `bands` is the demarkation between pass, transition, and stop bands.
    # `desired` indicates the desired gain at each point in `bands`
    # The combination of the two allows us to construct any type of filter!

    module_logger.info((f"design_pfb_fir: cut-off frequency: {freq_pass}, "
                        f"stop-band frequency: {freq_stop}, "
                        f"ntaps: {ntaps}"))

    bands = [0.0, freq_pass, freq_stop, 1.0]
    desired = [1, 1, 0, 0]
    weights = [1.0, 15.0]  # This is coming from Ian Morrion's matlab code

    coeff = scipy.signal.firls(ntaps, bands, desired, weight=weights)

    return coeff
