import cv2
import math
import h5py
import hdf5plugin
import numpy as np
from glob import glob
from typing import Dict, Tuple
from torchvision.io import decode_image
from torchvision.transforms.functional import pad


sequence_name = "zurich_city_14_c"

path_to_events = f"/local/richeek/dsec/test_events/{sequence_name}/events/left/events.h5"
path_to_semantic = f"/local/richeek/dsec/test_semantic_segmentation/test/{sequence_name}"

semantics_fp = f"{path_to_semantic}/11classes"
semantic_timestamps_fp = f"{path_to_semantic}/{sequence_name}_semantic_timestamps.txt"


def events_to_frames(events):
    evframe = np.zeros((480, 640, 3), dtype=np.uint8)
    evframe[events["y"], events["x"], 0] = 255 * (events["p"] == 1)  # ON events
    evframe[events["y"], events["x"], 2] = 255 * (events["p"] == 0)  # OFF events
    return evframe


# https://github.com/open-mmlab/mmsegmentation/blob/00790766aff22bd6470dbbd9e89ea40685008395/mmseg/utils/class_names.py#L249C1-L249C1
def cityscapes_palette(num_classes=19):
    """Cityscapes palette for external use."""
    palette = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
    palette = np.array(palette)
    if num_classes == 11:
        palette = palette[np.array([10, 2, 4, 11, 5, 0, 1, 8, 13, 3, 7])]
    return np.vstack((palette, np.array([0, 0, 0])))


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ["p", "x", "y", "t"]:
            self.events[dset_str] = self.h5f["events/{}".format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000, for ms > 0
        # (2) t[ms_to_idx[ms] - 1] < ms*1000, for ms > 0
        # (3) ms_to_idx[0] == 0
        # , where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f["ms_to_idx"], dtype="int64")

        if "t_offset" in list(h5f.keys()):
            self.t_offset = int(h5f["t_offset"][()])
        else:
            self.t_offset = 0
        self.t_final = int(self.events["t"][-1]) + self.t_offset

    def get_start_time_us(self):
        return self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset

        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(
            self.events["t"][t_start_ms_idx:t_end_ms_idx], dtype=np.uint64
        )
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(
            time_array_conservative, t_start_us, t_end_us
        )
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events["t"] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ["p", "x", "y"]:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events["t"].size
        return events

    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us / 1000)
        window_end_ms = math.ceil(ts_end_us / 1000)
        return window_start_ms, window_end_ms

    # @jit(nopython=True)
    @staticmethod
    def get_time_indices_offsets(
        time_array: np.ndarray, time_start_us: int, time_end_us: int
    ) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]


def main():
    semantics_files = sorted(
        glob(f"{semantics_fp}/*.png"), key=lambda x: int(x.split("/")[-1].split(".")[0])
    )
    semantics_timestamps = np.loadtxt(semantic_timestamps_fp, dtype=np.uint64)
    print(f"Found {len(semantics_files)} semantic files.")

    events_h5 = h5py.File(path_to_events, "r")
    t_offset = events_h5["t_offset"][()].astype(np.uint64)

    Δt = 50  # milliseconds
    eventslicer = EventSlicer(events_h5)
    for semantics_file, semantics_timestamp in zip(semantics_files, semantics_timestamps):
        print(f"Processing semantic file: {semantics_file}, timestamp: {semantics_timestamp}")
        semantics = decode_image(semantics_file).squeeze()  # 1, 440, 640 -> 440, 640
        semantics = pad(semantics, (0, 0, 0, 40), fill=11)  # pad bottom with 40 rows
        colored_semantics = cityscapes_palette(11)[semantics].astype(np.uint8)  # 480, 640, 3

        if semantics_timestamp < t_offset + Δt * 500:
            continue

        events_ = eventslicer.get_events(
            semantics_timestamp - Δt * 500, semantics_timestamp + Δt * 500
        )
        evframe = events_to_frames(events_)

        overlay_semantics = colored_semantics.copy()
        overlay_semantics[evframe.any(-1)] = 0.5 * overlay_semantics[evframe.any(-1)] + 128

        cv2.imshow("Events", evframe)
        cv2.imshow("Colored Semantics", colored_semantics)
        cv2.imshow("Overlay Semantics", overlay_semantics)
        key = cv2.waitKey(40)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
