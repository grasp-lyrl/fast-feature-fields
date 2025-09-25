import cv2
import h5py
import argparse
import numpy as np
from matplotlib import colormaps

from f3.tasks.depth.utils import get_depth_image, get_disparity_image

parser = argparse.ArgumentParser(description="Visualize M3ED depth data")
parser.add_argument(
    "--file", type=str, required=True, help="Path to the HDF5 file containing depth data"
)
parser.add_argument(
    "--dataset", type=str, default="m3ed", help="Dataset name to visualize depth data from"
)
parser.add_argument(
    "--depth_type", "-dt", type=str, choices=["depth", "disparity"], help="Type of depth data to visualize"
)
args = parser.parse_args()


def visualize_m3ed_depth(file_path):
    """Visualize depth data from M3ED dataset."""
    cmap = colormaps["jet"]
    with h5py.File(file_path, "r") as h5:
        depth_data = h5["depth/prophesee/left"]
        for idx in range(depth_data.shape[0]):
            depth = depth_data[idx]
            mask = (depth > 0) & (depth != np.inf)
            depth_image = get_depth_image(depth, mask, cmap)

            cv2.imshow("Depth Image", depth_image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


def visualize_m3ed_disparity(file_path):
    """Visualize depth data from M3ED dataset."""
    cmap = colormaps["plasma"]
    with h5py.File(file_path, "r") as h5:
        disp_path = h5["predictions"]
        for idx in range(disp_path.shape[0]):
            disp = disp_path[idx][..., 0]
            mask = (disp > 0) & (disp < 65535)
            disp_image = get_disparity_image(disp, mask, cmap)

            cv2.imshow("Disparity Image", disp_image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if args.dataset == "m3ed":
        if args.depth_type == "disparity":
            visualize_m3ed_disparity(args.file)
        else:
            visualize_m3ed_depth(args.file)
    else:
        print(f"Unknown dataset: {args.dataset}")
