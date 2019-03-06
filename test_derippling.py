# test_derippling.py
# This program attempts to assess the effectiveness of derippling on
# various filter lengths.
import os
import logging

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from pfb.pfb_channelizer import PFBChannelizer
from pfb.formats import DADAFile
import comparator.util
from comparator.multi_domain import TimeFreqDomainComparator

import src.generate_test_vectors as gen
import src.design_pfb_fir as des
import src.run_dspsr_with_dump as run
# import src.compare_dump_files as comp
# import src.util as util


def test_derippling():
    n_samples = 2**19
    fft_size = 229376

    os_factor = "8/7"
    nchan = 8
    ntaps_per_chan = [40]
    tsamp = 0.025
    # ntaps_per_chan = [5, 10, 20, 40]
    loc = 0.1
    freqs, phases = [int(loc*n_samples)], [np.pi/4.]
    offsets, widths = [int(loc*n_samples)], [1]

    comp = TimeFreqDomainComparator()
    comp.operators["diff"] = lambda a, b: np.abs(a - b)
    comp.operators["xcorr"] = lambda a, b: scipy.signal.fftconvolve(
        a, np.conj(b[::-1]), "full")
    comp.products["mean"] = np.mean
    comp.products["max"] = np.amax
    comp.products["power"] = lambda a: np.sum(a)**2  # not sure if its np.sum(a**2)
    comp.freq.set_fft_size(fft_size)
    comp.time.domain = [0, 1000]

    def tg():
        return gen.generate_time_domain_impulse(n_samples, offsets, widths)

    def fg():
        return gen.generate_complex_sinusoid(
            n_samples, freqs, phases, bin_offset=0.1)

    impulses = []
    impulse_files = []
    impulse_dump_files = []
    # impulse_names = ["freq", "time"]
    # impulse_funcs = [fg, tg]
    impulse_names = ["freq"]  #, "time"]
    impulse_funcs = [fg]  # , tg]
    for i in range(len(impulse_names)):
        impulse = impulse_funcs[i]()
        impulse_input = np.zeros((n_samples, 1, 2), dtype=impulse.dtype)
        impulse_input[:, 0, 0] = impulse
        impulse_input[:, 0, 1] = impulse
        impulse_file = DADAFile(f"{impulse_names[i]}_domain_impulse.dump")
        impulse_file.data = impulse_input
        impulse_file["TSAMP"] = tsamp
        impulse_file.dump_data()
        ar, dump = run.run_dspsr_with_dump(
            impulse_file.file_path,
            output_dir="./",
        )
        impulse_dump_files.append(dump)
        impulses.append(impulse_input)
        impulse_files.append(impulse_file)

    # with util.debug_plot(impulses, polar=False):
    #     pass
        # input(">> ")

    for ntaps in ntaps_per_chan:
        fir_coeff = des.design_pfb_fir(nchan*ntaps+1, nchan, os_factor)
        for i in range(len(impulses)):
            arr = impulses[i]
            impulse_dump_file_name = impulse_dump_files[i]
            print(impulse_dump_file_name)
            channelizer = PFBChannelizer(
                arr, fir_coeff, input_tsamp=0.025
            )
            channelizer.channelize(
                nchan, os_factor
            )
            out = channelizer.output_file_path
            ar, dump_dr = run.run_dspsr_with_dump(
                out,
                output_file_name=os.path.basename(out).replace(".dump", ".dr"),
                output_dir="./",
                extra_args="-IF 1:D -dr -V"
            )
            out = channelizer.output_file_path
            ar, dump_no_dr = run.run_dspsr_with_dump(
                out,
                # output_file_name=out.strip(".dump") + "dr"
                output_dir="./",
                extra_args="-IF 1:D -V"
            )

            dada_files = [DADAFile(f).load_data() for f in [
                dump_no_dr,
                dump_dr,
                impulse_dump_file_name
            ]]

            dada_files_pol0 = [d.data[:, 0, 0] for d in dada_files]

            time_res, freq_res = comp(*dada_files_pol0)
            comparator.util.corner_plot(time_res)
            comparator.util.corner_plot(freq_res)
            # comp.compare_dump_files(dump, dump_dr, op_str="sub")
            # print("Doing time domain comparison")
            # with comp.compare(
            #     DADAFile(impulse_dump_file_name).load_data(),
            #     DADAFile(dump_dr).load_data(),
            #     name="time_domain",
            #     fft_size=fft_size,
            #     n_samples=1.0
            # ) as plot_obj:
            #     comp.compare_time_domain(plot_obj)
            #
            # print("Doing frequency domain comparison")
            # with comp.compare(
            #     DADAFile(impulse_dump_file_name).load_data(),
            #     DADAFile(dump_dr).load_data(),
            #     name="freq_domain",
            #     fft_size=fft_size,
            #     n_samples=1.0
            # ) as plot_obj:
            #     comp.compare_freq_domain(plot_obj)
            # comp.compare_dump_files(
            #   dump, impulse_dump_file_name, op_str="sub", max_to_one=True)
            # comp.compare_dump_files(
            #   dump_dr, impulse_dump_file_name, op_str="sub", max_to_one=True)

        plt.show()
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    test_derippling()
