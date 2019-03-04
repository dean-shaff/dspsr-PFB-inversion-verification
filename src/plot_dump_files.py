# compare_dump_files.py
import argparse
import os

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from pfb_channelizer import PFBChannelizer

op_lookup = {
    "xcorr": lambda a, b: scipy.signal.fftconvolve(
        a, b[::-1], mode="same"
    ),
    "sub": lambda a, b: np.abs(np.subtract(a, b)),
    "div": lambda a, b: np.divide(a, b)
}


def plot_dump_files(*file_paths,
                    no_header=False,
                    complex=False,
                    n_samples=1.0,
                    op_str="xcorr"):
    if op_str not in op_lookup:
        raise KeyError(f"Can't find {op} in op_lookup")
    op = op_lookup[op_str]
    comp_dat = []
    dat_sizes = np.zeros(len(file_paths))
    header_offset = PFBChannelizer.header_size
    if no_header:
        header_offset = 0
    for i, fname in enumerate(file_paths):
        with open(fname, "rb") as input_file:
            buffer = input_file.read()
            # header = np.frombuffer(
            #     buffer, dtype='c', count=PFBChannelizer.header_size)
            dt = PFBChannelizer.input_dtype
            dt = np.dtype(dt).newbyteorder('=')
            data = np.frombuffer(
                buffer, dtype=dt,
                offset=header_offset)
            if complex:
                data = data.reshape((-1, 2))
                data = data[:, 0] + 1j*data[:, 1]
                data = np.abs(data)
                # data = data.reshape((2, -1))
                # data = data[0, :] + 1j*data[1, :]
            # data = data[:int(len(data)*n_samples)]

        dat_sizes[i] = data.shape[0]
        comp_dat.append(data)
    min_size = int(np.amin(dat_sizes) * n_samples)
    comp_dat = [dat[:min_size] for dat in comp_dat]
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
        # eighth = int(len(comp_dat[0]) / 8)
        # axes.plot(1.0/(comp_dat[0][:eighth]))
        # axes.axvline(int(len(comp_dat[0]) / 8), c="green")
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
