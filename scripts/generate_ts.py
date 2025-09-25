import h5py
import argparse
import hdf5plugin
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()

parser.add_argument("--data_h5", required=True, type=str, help="H5 file path with sensor data or the parent folder in case of dsec")
parser.add_argument("--bucket", type=int, default=20, help="Bucket size in us")
parser.add_argument("--dataset", type=str, default="m3ed", choices=["m3ed", "dsec", "mvsec"], help="Dataset name")

args = parser.parse_args()


def gen_ts(camera):
    if args.dataset == "m3ed":
        events_t = h5py.File(args.data_h5, 'r')[f'/prophesee/{camera}/t']
    elif args.dataset == "dsec":
        events_t = h5py.File(f'{args.data_h5}/events/{camera}/events.h5', 'r')['events/t']
    elif args.dataset == "mvsec":
        events_t = h5py.File(args.data_h5, 'r')[f'/davis/{camera}/events/t']
    else:
        raise ValueError("Invalid dataset")
    
    timeblocks = int(args.bucket)
    FREQ = 1e6/timeblocks
    num_ts = int(events_t[-1]/timeblocks)

    print("Frequency generated (in Hz): ", FREQ)
    print("Total time (in s): ", events_t[-1]/1e6)

    totaltime = events_t[-1]

    def return_index(start_index, till_when):
        start_event_index = start_index
        found_end = False
        counter = 0
        while not found_end:
            end_event_index = np.searchsorted(events_t[start_event_index+counter*1000:start_event_index+(counter+1)*1000], till_when)
            if end_event_index == 1000:
                counter += 1
            else:
                found_end = True
        end_event_index = start_event_index + counter*1000 + end_event_index
        return end_event_index

    TS = np.zeros((num_ts,), dtype=np.uint64)
    start_index = 0
    for block in tqdm(range(0, totaltime//timeblocks)):
        end_index = return_index(start_index, timeblocks*block)
        start_index = end_index
        TS[block] = end_index
    return TS


def main():
    if args.dataset == "m3ed":
        path = args.data_h5.rsplit("/", 1)[0]
        name = args.data_h5.split("/")[-1].replace(".h5", "")
    elif args.dataset == "dsec":
        path = args.data_h5
        name = args.data_h5.split("/")[-1]
    elif args.dataset == "mvsec":
        path = args.data_h5.rsplit("/", 1)[0]
        name = args.data_h5.split("/")[-1].replace(".hdf5", "")
    else:
        raise ValueError("Invalid dataset")

    TS_LEFT = gen_ts("left")
    TS_RIGHT = gen_ts("right")

    TS = {"left": TS_LEFT, "right": TS_RIGHT}
    np.save(f"{path}/50khz_{name}.npy", TS)


if __name__ == "__main__":
    main()

