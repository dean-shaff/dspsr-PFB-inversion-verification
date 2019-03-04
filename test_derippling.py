# test_derippling.py
# This program attempts to assess the effectiveness of derippling on
# various filter lengths.
import os
import logging

import numpy as np
from pfb.pfb_channelizer import PFBChannelizer

import src.generate_test_vectors as gen
import src.design_pfb_fir as des
import src.run_dspsr_with_dump as run
import src.compare_dump_files as comp


def test_derippling():

    n_samples = 2**20

    os_factor = "8/7"
    nchan = 8
    ntaps_per_chan = [5]
    # ntaps_per_chan = [5, 10, 20, 40]

    freqs, phases = [100], [np.pi/4.]
    offsets, widths = [100], [1]

    time_domain_impulse = gen.generate_time_domain_impulse(
        n_samples, offsets, widths)
    time_domain_input = np.zeros((n_samples, 1, 2),
                                 dtype=time_domain_impulse.dtype)
    time_domain_input[:, 0, 0] = time_domain_impulse
    time_domain_input[:, 0, 1] = time_domain_impulse
    freq_domain_impulse = gen.generate_complex_sinusoid(
        n_samples, freqs, phases, bin_offset=0.1)

    freq_domain_input = np.zeros((n_samples, 1, 2),
                                 dtype=freq_domain_impulse.dtype)
    freq_domain_input[:, 0, 0] = freq_domain_impulse
    freq_domain_input[:, 0, 1] = freq_domain_impulse

    for ntaps in ntaps_per_chan:
        fir_coeff = des.design_pfb_fir(nchan*ntaps+1, nchan, os_factor)
        channelizer_time_domain = PFBChannelizer(
            time_domain_input, fir_coeff, input_tsamp=0.025
        )
        channelizer_time_domain.channelize(
            nchan, os_factor
        )
        print(np.iscomplexobj(channelizer_time_domain.output_data))
        print(channelizer_time_domain.output_data.shape)
        print(channelizer_time_domain.output_data_file["NDIM"])
        ar, dump = run.run_dspsr_with_dump(
            channelizer_time_domain.output_file_path,
            "./", extra_args="-IF 1:D -dr"
        )
        dump_dr = f"{dump}.dr"
        os.rename(dump, dump_dr)
        ar, dump = run.run_dspsr_with_dump(
            channelizer_time_domain.output_file_path,
            "./", extra_args="-IF 1:D"
        )
        comp.compare_dump_files(dump, dump_dr, op_str="div")
        # os.remove(channelizer_time_domain.output_file_path)

        # channelizer_freq_domain = PFBChannelizer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_derippling()
