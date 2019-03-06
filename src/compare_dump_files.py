# compare_dump_files.py
from contextlib import contextmanager
import argparse
import os
import typing

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from pfb.formats import DADAFile

from .util import get_time_shift, get_nseries

op_lookup = {
    "xcorr": lambda a, b: scipy.signal.fftconvolve(
        a, b[::-1], mode="same"
    ),
    "sub": lambda a, b: np.abs(np.subtract(a, b)),
    "div": lambda a, b: np.divide(a, b)
}


@contextmanager
def compare(*dada_objs: typing.Tuple[DADAFile],
            name: str = "frequency",
            fft_size: int = 229376,
            n_samples: float = 1.0) -> None:

    fig, axes = plt.subplots(4, 4, figsize=(24, 24))

    nseries, min_size = get_nseries(*dada_objs, fft_size=fft_size)

    # nseries = int(nseries * n_samples)
    # xlim = [0, nseries*fft_size]

    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].grid(True)

    plot_obj = {
        "dada_objs": dada_objs,
        "fft_size": fft_size,
        "nseries": nseries,
        "n_samples": n_samples,
        "min_size": min_size,
        # "xlim": xlim,
        # "idx": slice(*xlim),
        "axes": axes,
        "ave_errors": [],
        "max_errors": []
    }

    try:
        yield plot_obj

    finally:
        fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        no_ext = [os.path.splitext(os.path.basename(obj.file_path))[0]
                  for obj in dada_objs]
        output_file_name = "validate_pfb_inversion.{}.{}.{}".format(
            name, *no_ext)
        fig.savefig(output_file_name + ".png")
        av, ma = plot_obj["ave_errors"], plot_obj["max_errors"]
        with open(output_file_name + ".txt", "w") as f:
            f.write("Ave, Max\n")
            for ipol in range(len(av)):
                for iz in range(len(av[ipol])):
                    output_str_list = [
                        f"{plot_obj['ave_errors'][ipol][iz]:.9f}",
                        f"{plot_obj['max_errors'][ipol][iz]:.9f}"
                    ]
                    f.write(", ".join(output_str_list) + "\n")
                    print("Average error ipol={}, z={}: {}, max error: {}".format(
                        ipol, iz, *output_str_list
                    ))


def compare_time_domain(plot_obj: dict) -> None:

    vanilla_dspsr_dump, pfb_inversion_dump = plot_obj["dada_objs"]
    axes = plot_obj["axes"]

    ndat_plot = int(plot_obj["n_samples"] * plot_obj["min_size"])

    for ipol in range(2):
        plot_obj["ave_errors"].append([])
        plot_obj["max_errors"].append([])

        for iz in range(2):
            pol_z_idx = ipol*2 + iz
            # z_func = np.abs if iz == 0 else np.angle
            # z_text = "Power spectrum" if iz == 0 else "Phase"
            z_func = np.real if iz == 0 else np.imag
            z_text = "Real" if iz == 0 else "Imaginary"

            van_dat = z_func(vanilla_dspsr_dump.data[100:ndat_plot, 0, ipol])
            van_dat /= np.amax(van_dat)
            inv_dat = z_func(pfb_inversion_dump.data[100:ndat_plot, 0, ipol])
            inv_dat /= np.amax(inv_dat)

            van = 0
            argmax = np.argmax(van_dat)
            axes[van, pol_z_idx].plot(van_dat, color="green")
            # axes[van].axvline(argmax, color="green")
            # axes[van, pol_z_idx].set_xlim([0, 1000])
            # axes[van].plot(vanilla_dspsr_dump.data[idx,0,0,1])
            axes[van, pol_z_idx].set_title(
                f"{z_text} Pol {ipol} Vanilla dspsr")

            inv = 1
            argmax = np.argmax(inv_dat)
            axes[inv, pol_z_idx].plot(inv_dat, color="green")
            # axes[inv, pol_z_idx].set_xlim([0, 1000])
            axes[inv, pol_z_idx].set_title(
                f"{z_text} Pol {ipol} PFB Inversion")

            # axes[inv].axvline(argmax, color="green")
            # axes[inv].plot(pfb_inversion_dump.data[idx,0,0,1])
            xcorr, offset = get_time_shift(van_dat, inv_dat)

            axes[inv + 1, pol_z_idx].plot(xcorr)
            # axes[inv + 1, pol_z_idx].set_xlim([0, 1000])

            axes[inv + 1, pol_z_idx].set_title(
                f"Cross Correlation: Offset={offset}")

            diff = np.roll(van_dat, abs(offset)) - inv_dat
            axes[inv + 2, pol_z_idx].set_yscale('log')
            # axes[inv + 2, pol_z_idx].set_xlim([0, 1000])
            axes[inv + 2, pol_z_idx].plot(np.abs(diff))
            axes[inv + 2, pol_z_idx].set_title(f"Offest corrected difference")
            plot_obj["ave_errors"][ipol].append(np.mean(np.abs(diff)))
            plot_obj["max_errors"][ipol].append(np.amax(np.abs(diff)))
            # print((f"Average error: {ave_errors[-1]:.9f}, "
            #        f"max error: {max_errors[-1]:.9f}"))


def compare_freq_domain(plot_obj: dict) -> None:

    vanilla_dspsr_dump, pfb_inversion_dump = plot_obj["dada_objs"]
    axes = plot_obj["axes"]
    fft_size = plot_obj["fft_size"]

    # z_lookup = {
    #     0: lambda x: np.abs(x),
    #     1: np.angle
    # }

    z_lookup = {
        0: np.real,
        1: np.imag
    }

    def log_abs(x):
        return np.log10(np.abs(x))

    for ipol in range(2):
        plot_obj["ave_errors"].append([])
        plot_obj["max_errors"].append([])
        for iz in range(2):
            pol_z_idx = ipol*2 + iz
            # z_func = np.abs if iz == 0 else np.angle
            # z_text = "Power spectrum" if iz == 0 else "Phase"
            z_func = z_lookup[iz]
            # z_text = "Power Spectrum" if iz == 0 else "Phase"
            # z_func = lambda x: np.log10(np.real(x)) if iz == 0 else lambda x: np.log10(np.imag(x))
            z_text = "Real" if iz == 0 else "Imaginary"

            van_dat = vanilla_dspsr_dump.data[fft_size:2*fft_size, 0, ipol]
            inv_dat = pfb_inversion_dump.data[fft_size:2*fft_size, 0, ipol]

            xcorr, offset = get_time_shift(van_dat, inv_dat)
            # print(f"offset={offset}")
            van_dat = np.roll(van_dat, abs(offset))

            van_dat_spec = z_func(np.fft.fft(van_dat))
            inv_dat_spec = z_func(np.fft.fft(inv_dat))
            van_dat_spec /= np.amax(van_dat_spec)
            inv_dat_spec /= np.amax(inv_dat_spec)

            van = 0
            argmax = np.argmax(van_dat_spec)
            axes[van, pol_z_idx].plot(van_dat_spec, color="green")
            axes[van, pol_z_idx].set_ylim([-0.1, 0.1])

            # axes[van, pol_z_idx].set_yscale('log')
            # axes[van].axvline(argmax, color="green")
            # axes[van].plot(vanilla_dspsr_dump.data[idx,0,0,1])
            axes[van, pol_z_idx].set_title(
                f"{z_text} Pol {ipol} Vanilla dspsr")

            inv = 1
            argmax = np.argmax(inv_dat_spec)
            # axes[inv, pol_z_idx].plot(log_abs(inv_dat_spec), color="green")
            axes[inv, pol_z_idx].plot(inv_dat_spec, color="green")
            axes[inv, pol_z_idx].set_ylim([-0.1, 0.1])
            # axes[inv, pol_z_idx].set_yscale('log')
            axes[inv, pol_z_idx].set_title(
                f"{z_text} Pol {ipol} PFB Inversion")


            diff = van_dat_spec - inv_dat_spec

            axes[inv + 1, pol_z_idx].set_yscale('log')
            axes[inv + 1, pol_z_idx].plot(np.abs(diff))
            axes[inv + 1, pol_z_idx].set_title(f"Offest corrected difference")
            plot_obj["ave_errors"][ipol].append(np.mean(np.abs(diff)))
            plot_obj["max_errors"][ipol].append(np.amax(np.abs(diff)))


def compare_dump_files(*file_paths: typing.Tuple[str],
                       n_samples: float = 1.0,
                       op_str: str = "sub",
                       max_to_one: bool = False):
    if op_str not in op_lookup:
        raise KeyError(f"Can't find {op} in op_lookup")
    op = op_lookup[op_str]
    comp_dat = []
    dat_sizes = np.zeros(len(file_paths))
    dada_files = [DADAFile(f) for f in file_paths]
    for i in range(len(dada_files)):
        dada_files[i].load_data()
        dat_sizes[i] = dada_files[i].ndat

    for key in ["NCHAN", "NPOL", "NDIM"]:
        if len(set([d[key] for d in dada_files])) > 1:
            raise RuntimeError(
                ("Need to make sure number of channels and "
                 "polarizations is the same in order to compare dada files"))
    min_size = int(np.amin(dat_sizes) * n_samples)
    comp_dat = [d.data[:min_size, :, :].flatten() for d in dada_files]
    if max_to_one:
        comp_dat = [d / np.amax(np.abs(d)) for d in comp_dat]
    iscomplex = all([np.iscomplexobj(arr) for arr in comp_dat])
    if iscomplex:
        for i in range(len(comp_dat)):
            data = comp_dat[i]
            comp_dat[i] = np.zeros(2*data.shape[0])
            comp_dat[i][::2] = data.real
            comp_dat[i][1::2] = data.imag

    subplot_dim = [len(file_paths) for i in range(2)]
    fig_size_mul = [6, 4]
    fig, axes = plt.subplots(
        *subplot_dim,
        figsize=tuple([i*j for i, j in zip(subplot_dim, fig_size_mul)]))
    fig.tight_layout()

    # for i in range(min_size):
    #     print(comp_dat[0][i], comp_dat[1][i])

    if not hasattr(axes, "__iter__"):
        axes.grid(True)
        axes.plot(comp_dat[0])
        axes.set_title(os.path.basename(file_paths[0]))
    else:
        for row in range(axes.shape[0]):
            for col in range(axes.shape[1]):
                ax = axes[row, col]
                ax.grid(True)
                if row == col:
                    ax.grid(True)
                    ax.plot(comp_dat[row])
                    ax.set_title(os.path.basename(file_paths[row]))
                elif row > col:
                    res = op(comp_dat[row], comp_dat[col])
                    ax.plot(res)
                    ax.set_title(
                        (f"{op_str}: "
                         f"{os.path.basename(file_paths[row])}, "
                         f"{os.path.basename(file_paths[col])}"))
                else:
                    ax.set_axis_off()
    plt.show()


def create_parser():

    parser = argparse.ArgumentParser(
        description="compare the contents of two dump files")

    parser.add_argument("-i", "--input-files",
                        dest="input_file_paths",
                        nargs="+", type=str,
                        required=True)

    parser.add_argument('-nh', "--no-header",
                        dest="no_header", action="store_true")

    parser.add_argument('-c', "--complex",
                        dest="complex", action="store_true")

    parser.add_argument('-n', "--n-samples",
                        dest="n_samples", type=float)

    parser.add_argument(
        '-op', "--operation",
        dest="op", type=str,
        help=(f"Apply 'op' to each pair of input files. "
              f"Available operations: {list(op_lookup.keys())}"))

    parser.add_argument("-v", "--verbose",
                        dest="verbose", action="store_true")

    return parser


if __name__ == '__main__':
    parsed = create_parser().parse_args()
    plot_dump_files(
        *parsed.input_file_paths,
        no_header=parsed.no_header,
        complex=parsed.complex,
        n_samples=parsed.n_samples,
        op_str=parsed.op
    )
