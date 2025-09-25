import cv2
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm

from f3.utils import BaseExtractor, ev_to_frames

parser = argparse.ArgumentParser(description="Visualize M3ED data with context frames.")
parser.add_argument("--h5file", type=str, required=True, help="Path to the HDF5 file containing M3ED data.")
parser.add_argument("--ts50khz", type=str, required=True, help="Path to the 50kHz timestamp file.")
args = parser.parse_args()

h5file = args.h5file
ts50khz = args.ts50khz
h5loaded = h5py.File(h5file, "r")

extractor = BaseExtractor(h5file, ts50khz, w=1280, h=720, time_ctx=20000,
                          time_pred=20000, bucket=1000, max_numevents_ctx=800000,
                          randomize_ctx=False, camera="left")

start_idx = 50

ovc_load = "rgb"
if "ovc/rgb/data" not in h5loaded:
    print("No RGB data found in the HDF5 file. Using 'left' context instead.")
    ovc_load = "left"

end_idx = len(h5loaded[f"ovc/{ovc_load}/data"])
step = 1

for rgb_idx_ in tqdm(range(start_idx, end_idx, step)):
    timestamp = h5loaded["ovc/ts"][rgb_idx_]  # in us
    rgb = h5loaded[f"ovc/{ovc_load}/data"][rgb_idx_][:720].squeeze()  # (H, W, C)
    ctx, totcnt = extractor.get_ctx_fixedtime(timestamp)
    
    ctx_frame = ev_to_frames(ctx, torch.tensor([totcnt]), w=1280, h=720)
    ctx_frame = ctx_frame[0].cpu().numpy().T  # (H, W)
    
    # convert context frame to BGR (from grayscale) for stacking
    if ovc_load == "rgb":
       ctx_frame = cv2.cvtColor(ctx_frame, cv2.COLOR_GRAY2BGR)
    # stack both frames side-by-side
    combined = np.hstack((rgb, ctx_frame))    
    cv2.putText(combined, f"Timestamp: {timestamp} us", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(combined, f"Total Events: {totcnt}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("RGB (left) | Context (right)", combined)
    
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

