import h5py
import argparse
import hdf5plugin
import numpy as np

parser = argparse.ArgumentParser(description='Split DAVIS hdf5 file into x, y, t, p')
parser.add_argument('input_file', type=str, help='Path to input hdf5 file')
parser.add_argument('output_file', type=str, help='Path to output hdf5 file')
args = parser.parse_args()


def split_hdf5(input_file, output_file):
    basel = f"/davis/left/events"
    baser = f"/davis/right/events"

    with h5py.File(output_file, 'w') as f_out:
        with h5py.File(input_file, 'r+') as f:
            data = f[basel]
            x, y, t, p = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            absolute_start_time = t[0]
            t = ((t - absolute_start_time) * 1e6).astype(np.uint64)
        f_out.create_dataset(f"{basel}/x", data=x, chunks=(40000,), dtype=np.uint16, compression="lzf")
        f_out.create_dataset(f"{basel}/y", data=y, chunks=(40000,), dtype=np.uint16, compression="lzf")
        f_out.create_dataset(f"{basel}/t", data=t, chunks=(40000,), dtype=np.uint64, compression="lzf")
        f_out.create_dataset(f"{basel}/p", data=p, chunks=(40000,), dtype=np.int8, compression="lzf")
        f_out.attrs["absolute_start_time"] = absolute_start_time

        with h5py.File(input_file, 'r+') as f:
            data = f[baser]
            x, y, t, p = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            absolute_start_time = t[0]
            t = ((t - absolute_start_time) * 1e6).astype(np.uint64)
        f_out.create_dataset(f"{baser}/x", data=x, chunks=(40000,), dtype=np.uint16, compression="lzf")
        f_out.create_dataset(f"{baser}/y", data=y, chunks=(40000,), dtype=np.uint16, compression="lzf")
        f_out.create_dataset(f"{baser}/t", data=t, chunks=(40000,), dtype=np.uint64, compression="lzf")
        f_out.create_dataset(f"{baser}/p", data=p, chunks=(40000,), dtype=np.int8, compression="lzf")
        f_out.attrs["absolute_start_time"] = absolute_start_time


if __name__ == "__main__":
    split_hdf5(args.input_file, args.output_file)
