# test_derippling.py
# This program attempts to assess the effectiveness of derippling on
# various filter lengths.
import os
import sys
import logging
import typing

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from pfb.pfb_channelizer import PFBChannelizer
from pfb.formats import DADAFile
from pfb.rational import Rational
import comparator.util
from comparator.multi_domain import TimeFreqDomainComparator

import src.generate_test_vectors as gen
import src.design_pfb_fir as des
import src.run_dspsr_with_dump as run


def test_derippling(
    n_samples: int,
    fft_size: int,
    n_taps_per_chan: typing.List[int],
    oversampling_factor: str,
    n_pfb_chan: int,
    tsamp: float,
    impulse_gen_callable: typing.Callable,
    impulse_name: str
) -> None:

    def diff(a, b):
        abs_diff = np.abs(a/np.amax(a) - b/np.amax(b))
        return abs_diff

    comp = TimeFreqDomainComparator()
    comp.operators["diff"] = diff
    # comp.operators["xcorr"] = lambda a, b: scipy.signal.fftconvolve(
    #     a, np.conj(b[::-1]), "full")
    comp.products["mean"] = np.mean
    comp.products["std"] = np.std
    comp.products["max"] = np.amax
    comp.products["max_arg"] = np.argmax
    comp.products["power"] = lambda a: np.sum(a**2)
    comp.products["rms"] = lambda a: np.sqrt(np.mean(a**2))
    comp.freq.domain = [0, fft_size]
    comp.time.domain = [0, 1000]

    impulse = impulse_gen_callable()
    impulse_input = np.zeros((n_samples, 1, 2), dtype=impulse.dtype)
    impulse_input[:, 0, 0] = impulse
    impulse_input[:, 0, 1] = impulse
    impulse_file = DADAFile(f"{impulse_name}_domain_impulse.dump")
    impulse_file.data = impulse_input
    impulse_file["TSAMP"] = tsamp
    impulse_file.dump_data()
    ar, impulse_dump_file = run.run_dspsr_with_dump(
        impulse_file.file_path,
        output_dir="./",
        extra_args=f"-x {fft_size}"
    )
    os_factor_rational = Rational(*oversampling_factor.split("/"))
    fft_size_norm = os_factor_rational.normalize(fft_size)
    for ntaps in n_taps_per_chan:
        fir_coeff = des.design_pfb_fir(
            n_pfb_chan*ntaps+1, n_pfb_chan, oversampling_factor)
        channelizer = PFBChannelizer(
            impulse_input, fir_coeff, input_tsamp=0.025
        )
        channelizer.channelize(
            n_pfb_chan, oversampling_factor
        )
        out = channelizer.output_file_path
        ar, dump_dr = run.run_dspsr_with_dump(
            out,
            output_file_name=os.path.basename(out).replace(".dump", ".dr"),
            output_dir="./products",
            extra_args=f"-IF 1:D -dr -V"
        )
        out = channelizer.output_file_path
        ar, dump_no_dr = run.run_dspsr_with_dump(
            out,
            # output_file_name=out.strip(".dump") + "dr"
            output_dir="./products",
            extra_args=f"-IF 1:D -V"
        )

        dada_files = [DADAFile(f).load_data() for f in [
            dump_no_dr,
            dump_dr,
            impulse_dump_file
        ]]

        dada_files_pol0 = [d.data[:, 0, 0] for d in dada_files]
        dada_files_pol0[0] /= fft_size_norm
        dada_files_pol0[1] /= fft_size_norm

        # time_res = comp.time(*dada_files_pol0)
        freq_res_op, freq_res_prod = comp.freq.polar(*dada_files_pol0)
        for i, row in enumerate(freq_res_prod["diff"]):
            for j, col in enumerate(row):
                if col is not None:
                    print(f"{i}, {j}: {col[-1]['mean']}")
        # comparator.util.corner_plot(time_res)
        comparator.util.corner_plot(freq_res_op)

    plt.show()


def main():
    fft_size = 229376
    # fft_size = 32768
    # n_samples = fft_size*2
    n_samples = 2**19
    # os_factor = "8/7"
    os_factor = "1/1"
    n_chan = 8
    n_taps_per_chan = [10]
    tsamp = 0.025

    loc = 0.1
    freqs, phases = [int(loc*n_samples)], [np.pi/4.]
    offsets, widths = [int(loc*n_samples)], [1]

    print(loc * fft_size)

    def tg():
        return gen.generate_time_domain_impulse(n_samples, offsets, widths)

    def fg():
        return gen.generate_complex_sinusoid(
            n_samples, freqs, phases, bin_offset=0.1)

    test_derippling(
        n_samples,
        fft_size,
        n_taps_per_chan,
        os_factor,
        n_chan,
        tsamp, fg,
        "frequency"
    )
    # test_derippling(
    #     n_samples,
    #     fft_size,
    #     n_taps_per_chan,
    #     os_factor,
    #     n_chan,
    #     tsamp, tg,
    #     "time")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    main()
