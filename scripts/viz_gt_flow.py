import cv2
import h5py
import argparse
import numpy as np

from f3.tasks.optical_flow.utils import flow_viz_np

parser = argparse.ArgumentParser(description="Visualize M3ED/MVSEC flow data")
parser.add_argument(
    "--file", type=str, required=True, help="Path to the HDF5 file containing flow data"
)
parser.add_argument(
    "--dataset", type=str, default="mvsec", help="Dataset name to visualize flow data from"
)
args = parser.parse_args()


def visualize_mvsec_flow(file_path):
    """Visualize flow data from MVSEC dataset."""
    with h5py.File(file_path, "r") as h5:
        image_path = "davis/left/blended_image_rect"
        flow_path = "davis/left/flow_dist"
        flow_data, image_data = h5[flow_path], h5[image_path]

        for idx in range(flow_data.shape[0]):
            flow = flow_data[idx].transpose(1, 2, 0)
            flow_rgb = flow_viz_np(flow)
            image = image_data[idx]

            cv2.imshow("flow", flow_rgb)
            cv2.imshow("image", image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


def visualize_m3ed_flow(file_path):
    """Visualize flow data from M3ED dataset."""
    with h5py.File(file_path, "r") as h5:
        flow_x_path = "flow/prophesee/left/x"
        flow_y_path = "flow/prophesee/left/y"
        flow_data_x = h5[flow_x_path]
        flow_data_y = h5[flow_y_path]

        for idx in range(flow_data_x.shape[0]):
            flow = np.stack((flow_data_x[idx], flow_data_y[idx]), axis=-1)
            flow_rgb = flow_viz_np(flow, norm=True) # use norm if you want better colors

            cv2.imshow("flow", flow_rgb)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


# Main execution
if args.dataset == "mvsec":
    visualize_mvsec_flow(args.file)
elif args.dataset == "m3ed":
    visualize_m3ed_flow(args.file)
else:
    print(f"Unknown dataset: {args.dataset}")
