import os
import h5py
import argparse
import hdf5plugin
from collections import defaultdict

args = argparse.ArgumentParser(
    description="Dataset statistics",
    usage="Example: python dataset_stats.py -p data/spot*",
)
args.add_argument("-p", "--paths", type=str, nargs="+", help="Paths to the dataset directories")
args.add_argument(
    "-d",
    "--data_type",
    type=str,
    default="M3ED",
    choices=["M3ED", "DSEC", "MVSEC"],
    help="Type of dataset to process (default: M3ED)",
)
args = args.parse_args()


def get_m3ed_events_and_images_stats(h5_file):
    with h5py.File(h5_file, "r") as f:
        has_rgb = "ovc/rgb/data" in f
        ovc_left_cnt = f["ovc/left/data"].shape[0]
        left_ev_cnt = f["prophesee/left/t"].shape[0]
        total_evtime = (
            f["prophesee/left/t"][-1] - f["prophesee/left/t"][0]
        ) / 1e6  # Convert to seconds
        print(f"File: {h5_file}")
        print(f"  File has RGB: {has_rgb}")
        if has_rgb:
            ovc_rgb_cnt = f["ovc/rgb/data"].shape[0]
            print(f"  OVC RGB frames: {ovc_rgb_cnt}")
        print(f"  OVC Left Grayscale frames: {ovc_left_cnt}")
        print(f"  Prophesee left events: {left_ev_cnt}")
        print(f"  Total event time: {total_evtime:.2f} seconds")
        print("#" + "-" * 40 + "# \n")

        return {
            "has_rgb": has_rgb,
            "ovc_left_cnt": ovc_left_cnt,
            "ovc_rgb_cnt": ovc_rgb_cnt if has_rgb else 0,
            "left_ev_cnt": left_ev_cnt,
            "total_evtime": total_evtime,
        }


def get_m3ed_semantics_stats(h5_file):
    with h5py.File(h5_file, "r") as f:
        print(f"File: {h5_file}")
        semantics_cnt = f["predictions"].shape[0]
        print(f"  Semantics frames: {semantics_cnt}")
        print("#" + "-" * 40 + "# \n")
        return semantics_cnt


def get_m3ed_depth_stats(h5_file):
    with h5py.File(h5_file, "r") as f:
        print(f"File: {h5_file}")
        depth_cnt = f["depth/prophesee/left"].shape[0]
        print(f"  Depth frames: {depth_cnt}")
        print("#" + "-" * 40 + "# \n")
        return depth_cnt


def get_dsec_events_stats(ev_file):
    with h5py.File(ev_file, "r") as f:
        print(f"File: {ev_file}")
        evcnt = f["events/t"].shape[0]
        total_evtime = (f["events/t"][-1] - f["events/t"][0]) / 1e6  # Convert to seconds
        print(f"  Events: {evcnt}")
        print(f"  Total event time: {total_evtime:.2f} seconds")
        print("#" + "-" * 40 + "# \n")
        return {
            "evcnt": evcnt,
            "total_evtime": total_evtime,
        }


def get_mvsec_events_stats(h5_file):
    with h5py.File(h5_file, "r") as f:
        print(f"File: {h5_file}")
        evcnt = f["davis/left/events/t"].shape[0]
        total_evtime = (f["davis/left/events/t"][-1] - f["davis/left/events/t"][0]) / 1e6
        print(f"  Events: {evcnt}")
        print(f"  Total event time: {total_evtime:.2f} seconds")
        print("#" + "-" * 40 + "# \n")

        return {
            "evcnt": evcnt,
            "total_evtime": total_evtime,
        }


if __name__ == "__main__":
    cumulative_stats = defaultdict(float)

    for folder in args.paths:
        name = os.path.basename(folder)

        if args.data_type == "M3ED":
            if name.endswith("_data"):
                name = name[:-5]
            event_and_images_file = folder + f"/{name}_data.h5"
            depth_gt_file = folder + f"/{name}_depth_gt.h5"
            semantics_gt_file = folder + f"/{name}_semantics.h5"

            if os.path.exists(event_and_images_file):
                stats = get_m3ed_events_and_images_stats(event_and_images_file)
                for key, value in stats.items():
                    cumulative_stats[key] += value

            if os.path.exists(semantics_gt_file):
                if stats.get("has_rgb", False):
                    cumulative_stats["semantics_cnt"] += get_m3ed_semantics_stats(semantics_gt_file)

            if os.path.exists(depth_gt_file):
                cumulative_stats["depth_cnt"] += get_m3ed_depth_stats(depth_gt_file)

        elif args.data_type == "DSEC":
            left_ev_file = folder + "/events/left/events.h5"
            # right_ev_file = folder + "/events/right/events.h5"
            left_im_folder = folder + "/images/left/rectified"
            # right_im_folder = folder + "/images/right/rectified"
            disparity_folder = folder + "/disparity/disparity/event"
            semantics_folder = folder + "/segmentation/11classes"

            if os.path.exists(left_ev_file):
                stats = get_dsec_events_stats(left_ev_file)
                for key, value in stats.items():
                    cumulative_stats[key] += value
            if os.path.exists(left_im_folder):
                cumulative_stats["left_im_cnt"] += len(os.listdir(left_im_folder))
            if os.path.exists(disparity_folder):
                cumulative_stats["disparity_cnt"] += len(os.listdir(disparity_folder))
            if os.path.exists(semantics_folder):
                cumulative_stats["semantics_cnt"] += len(os.listdir(semantics_folder))

        elif args.data_type == "MVSEC":
            if name.endswith("_data"):
                name = name[:-5]
            events_file = folder + f"/{name}_data.hdf5"
            depth_and_flow_file = folder + f"/{name}_gt.hdf5"

            if os.path.exists(events_file):
                stats = get_mvsec_events_stats(events_file)
                for key, value in stats.items():
                    cumulative_stats[key] += value
            if os.path.exists(depth_and_flow_file):
                with h5py.File(depth_and_flow_file, "r") as f:
                    cumulative_stats["depth_cnt"] += f["davis/left/depth_image_raw"].shape[0]

    print(f"Cumulative stats for all datasets:")
    for key, value in cumulative_stats.items():
        print(f"  {key}: {value}")


"""
M3ED
-----
python3 scripts/dataset_stats.py -p data/car_urban_day*                                                               
python3 scripts/dataset_stats.py -p data/car_urban_night*                                                             
python3 scripts/dataset_stats.py -p data/falcon_*                                                                     
python3 scripts/dataset_stats.py -p data/spot_*                                                                       

DSEC
-----
python3 scripts/dataset_stats.py -p data/zurich_* -d DSEC                                                             
python3 scripts/dataset_stats.py -p data/thun_* -d DSEC                                                               
python3 scripts/dataset_stats.py -p data/interlaken_* -d DSEC      

MVSEC
-----
"""
