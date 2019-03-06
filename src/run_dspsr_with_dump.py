# run_dspsr_with_dump.py
import os
import logging
import subprocess
import argparse
import shlex
import typing

__all__ = [
    "run_dspsr_with_dump"
]

module_logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")


def run_dspsr_with_dump(file_path: str,
                        output_dir: str = None,
                        output_file_name: str = None,
                        extra_args: str = None) -> typing.Tuple[str]:

    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    if output_file_name is None:
        file_name_base = os.path.splitext(file_name)[0]
    else:
        file_name_base = output_file_name

    if output_dir is None:
        output_dir = file_dir

    if extra_args is None:
        extra_args = ""

    output_ar = os.path.join(output_dir, file_name_base)
    output_dump = os.path.join(
        output_dir, f"pre_Detection.{file_name_base}.dump")
    output_log = os.path.join(
        output_dir, f"{file_name_base}.log")

    module_logger.debug(f"run_dspsr_with_dump: output archive: {output_ar}")
    module_logger.debug(f"run_dspsr_with_dump: output dump: {output_dump}")
    module_logger.debug(f"run_dspsr_with_dump: output log: {output_log}")

    dspsr_cmd_str = (f"dspsr -c 0.00575745 -D 2.64476 {file_path} "
                     f"-O {output_ar} -dump Detection {extra_args}")

    module_logger.debug(f"run_dspsr_with_dump: dspsr command: {dspsr_cmd_str}")

    after_cmd_str = f"mv pre_Detection.dump {output_dump}"

    cleanup_cmd_str = "rm *.dat"

    try:
        with open(output_log, "w") as log_file:
            dspsr_cmd = subprocess.run(shlex.split(dspsr_cmd_str),
                                       stdout=log_file,
                                       stderr=log_file)
        if dspsr_cmd.returncode == 0:
            subprocess.run(shlex.split(after_cmd_str))
    except subprocess.CalledProcessError as err:
        print(f"Couldn't execute command {dspsr_cmd_str}: {err}")
    finally:
        subprocess.run(cleanup_cmd_str, shell=True)

    return (f"{output_ar}.ar", output_dump)


def create_parser():
    parser = argparse.ArgumentParser(
        description="run dspsr")

    parser.add_argument("-i", "--input-file",
                        dest="input_file_path",
                        required=True)

    parser.add_argument("-a", "--args",
                        dest="extra_args",
                        required=False)

    return parser


if __name__ == '__main__':
    parsed = create_parser().parse_args()
    run_dspsr_with_dump(
        parsed.input_file_path,
        output_dir=data_dir,
        extra_args=parsed.extra_args
    )
