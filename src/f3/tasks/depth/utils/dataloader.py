import cv2
import h5py
import yaml
import logging
import hdf5plugin
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from f3.utils import EventDatasetSingleHDF5, collate_fn_general


#! Master Dataloader
class TimeAlignedDepthAndEvents(EventDatasetSingleHDF5):
    """
        Loads the raw M3ED dataset from the hdf5 file and samples the negative events
        on the fly. This code ignores the duplication of events caused by bucketing. 
        This is mostly tolerable for >1kHz frames.
    """
    def __init__(self, hdf5_file: str, timestamps_50khz_file: str, depth_path: str,
                 min_numevents_ctx: int=200000, max_numevents_ctx: int=800000,
                 time_ctx: int=15000, bucket: int=1000, w: int=1280, h: int=720,
                 randomize_ctx: bool=True, camera: str="left", dtype: str="m3ed_pseudo"):
        """
            Args:
                depth_hdf5_file: str
                    Path to the hdf5 file containing the depth predictions
                
                dtype: "m3ed_pseudo", "m3ed_gt" or "dsec_gt"
        """
        if dtype == "m3ed_pseudo": # DepthAnything monocular depth remapped to event frame
            self.depths = h5py.File(depth_path, "r")
            self.disparity = self.depths['predictions'] # 16bit disparity maps from depth anything v2
            self.ts_map = self.depths['ts'][:]
            self.src_ofst_res = torch.tensor([(h - 720) // 2, (w - 1280) // 2, 720, 1280], dtype=torch.int32)

        elif dtype == "m3ed_gt": # True LIDAR depth, remapped to event frame
            self.fb = 10.315 * 12 # focal length * baseline in pixel * meters
            self.depths = h5py.File(depth_path, "r")
            self.depth = self.depths['depth/prophesee/left'] # float32 depth maps from LIDAR in prophesee left frame
            self.ts_map = self.depths[f'ts'][:].astype(np.uint64)
            self.src_ofst_res = torch.tensor([(h - 720) // 2, (w - 1280) // 2, 720, 1280], dtype=torch.int32)

        elif dtype == "dsec_gt": # True depth as disparity maps, mapped to event frame
            folder = Path(depth_path)
            png_paths = sorted(
                (folder / "disparity" / "disparity" / "event").glob("*.png"),
                key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or 0)
            )
            self.disparity_paths = [str(path) for path in png_paths]
            timestamps_path = folder / "disparity" / "disparity" / "timestamps.txt"
            self.ts_map = (np.loadtxt(timestamps_path, dtype=np.uint64) - h5py.File(hdf5_file, "r")['t_offset'][()]).astype(np.uint64)
            self.src_ofst_res = torch.tensor([(h - 480) // 2, (w - 640) // 2, 480, 640], dtype=torch.int32)

        elif dtype == "mvsec_gt":
            self.fb = 10.315 * 10 # focal length * baseline in pixel * meters
            self.depth_h5 = h5py.File(depth_path, "r")
            self.depth = self.depth_h5['davis/left/depth_image_raw']
            self.ts_map = ((
                self.depth_h5['davis/left/depth_image_raw_ts'][:] -\
                h5py.File(hdf5_file, "r").attrs["absolute_start_time"]
            ) * 1e6).astype(np.uint64)
            self.src_ofst_res = torch.tensor([(h - 260) // 2, (w - 346) // 2, 260, 346], dtype=torch.int32)

        elif dtype == "tartanair-v2_gt":
            self.fb = 320 * 0.25 # focal length * baseline in pixel * meters
            folder = Path(depth_path)
            png_paths = sorted(
                (folder / f"depth_{camera}").glob("*.png"),
                key=lambda x: int(x.stem.split('_')[0])
            )
            self.depth = [str(path) for path in png_paths]
            if len(self.depth) == 0:
                raise ValueError(f"No depth png files found in {folder / f'depth_{camera}'}!")
            self.ts_map = np.arange(len(self.depth), dtype=np.uint64) * 100000  # Tartanair-v2 depth is at 10Hz, so every 100000us
            self.src_ofst_res = torch.tensor([(h - 640) // 2, (w - 640) // 2, 640, 640], dtype=torch.int32)

        else:
            raise ValueError("dtype should be either m3ed or dsec!")

        self.transform = lambda disparity: cv2.resize(disparity, [h, w], interpolation=cv2.INTER_NEAREST)

        dtype, self.mode = dtype.split('_') # [m3ed, dsec, tartanair-v2]x[pseudo, gt]
        super(TimeAlignedDepthAndEvents, self).__init__(
            hdf5_file=hdf5_file, timestamps_50khz_file=timestamps_50khz_file, w=w, h=h,
            min_numevents_ctx=min_numevents_ctx, max_numevents_ctx=max_numevents_ctx,
            time_ctx=time_ctx, bucket=bucket, randomize_ctx=randomize_ctx, camera=camera, dtype=dtype
        )
        self.logger.info("### DEPTH DATALOADER ###")

    def process_metadata(self):
        if self.load_metadata(keys=["camera", "dtype", "mode", "min_numevents_ctx", "time_ctx"], fname="metadata_depths.json"):
            self.logger.info(f"Dataset {Path(self.hdf5_fp).name} + {self.camera} loaded successfully!: {self.numblocks} valid blocks found!")
        else:
            self.valid_0_points = []
            for idx in tqdm(range(self.ts_map.size), "Metadata Depths"):
                t0 = self.ts_map[idx] + self.time_ctx // 2
                if (t0 <= self.time_ctx) or (t0 // self.us_to_discretize >= self.timestamps.shape[0]):
                    continue
                cnt = self.timestamps[t0 // self.us_to_discretize] - \
                      self.timestamps[(t0 - self.time_ctx) // self.us_to_discretize]
                if cnt >= self.min_numevents_ctx:
                    self.valid_0_points.append(idx)
                    self.logger.info(f"Valid index: {idx}!")
            self.numblocks = len(self.valid_0_points) # number of data points we have for training and testing
            self.logger.info(f"Dataset {Path(self.hdf5_fp).name} loaded successfully!: {self.numblocks} valid blocks found!")
            self.metadata = {
                "camera": self.camera,
                "dtype": self.dtype,
                "mode": self.mode,
                "min_numevents_ctx": self.min_numevents_ctx,
                "time_ctx": self.time_ctx,
                "valid_0_points": self.valid_0_points
            }
            self.save_metadata(fname="metadata_depths.json")

    def __getitem__(self, idx):
        fidx = self.valid_0_points[idx]
        t0 = self.ts_map[fidx] + self.time_ctx // 2
        ctx, totcnt = self.get_ctx_fixedtime(t0) # Load from [-time_ctx//2 ... t0 ... time_ctx//2]

        if self.dtype == "m3ed":
            if self.mode == "pseudo":
                disparity = self.disparity[fidx].squeeze(-1) # 720x1280x1 -> 720x1280
            elif self.mode == "gt":
                # mask out anything above 200m (m3ed gt is upto 120m, so essentially we just want to mask out inf)
                badmask = self.depth[fidx] > 200
                disparity = self.fb / self.depth[fidx] # 720x1280 focal length * baseline
                disparity[badmask] = 65535

        elif self.dtype == "dsec":
            #! DSEC GT disparity is multiplied by 256 and saved as 16bit png
            disparity = cv2.imread(self.disparity_paths[fidx], cv2.IMREAD_ANYDEPTH) / 256. # 480x640
            disparity[disparity == 0] = 65535

        elif self.dtype == "mvsec":
            badmask = np.isnan(self.depth[fidx]) | (self.depth[fidx] > 80) | (self.depth[fidx] < 0.1)
            disparity = self.fb / self.depth[fidx]
            disparity[badmask] = 65535

        elif self.dtype == "tartanair-v2":
            #! Tartanair-v2 depth is in 4 channel png files, need to read and convert to float32 depth in meters
            depth = cv2.imread(self.depth[fidx], cv2.IMREAD_UNCHANGED)
            depth = depth.view("<f4").squeeze(-1)
            badmask = np.isnan(depth) | (depth > 400) | (depth < 0.01) # Just a sanity filter. Implement actual filtering in train loop.
            disparity = self.fb / depth # So should produce disparity in range of [0.2, 8000] for depths [0.01, 400] meters
            disparity[badmask] = 65535

        disparity = np.pad(disparity, ((self.src_ofst_res[0], self.src_ofst_res[0]),
                                       (self.src_ofst_res[1], self.src_ofst_res[1])),
                           mode='constant', constant_values=65535)
        return ctx, totcnt, disparity, self.src_ofst_res


def get_dataset_from_h5files(hdf5_files: list[str], timestamps_files: list[str], depth_paths: list[str],
                             cameras: list[str], dtypes: list[str], ranges: list[list[float, int, float]],
                             **kwargs):
    assert len(hdf5_files) == len(timestamps_files), "The number of hdf5 files and timestamps files should be the same!"
    if cameras is None:
        cameras = ["left"] * len(hdf5_files)   # default to left camera
    if ranges is None:
        ranges = [[0, 1, 1]] * len(hdf5_files) # default to the whole dataset start, step, stop -> start and stop are fractions
    if dtypes is None:
        dtypes = ["m3ed"] * len(hdf5_files)    # default to m3ed dataset
    
    datasets = [
        TimeAlignedDepthAndEvents(hdf5_file, timestamps_file, depth_path, camera=camera, dtype=dtype, **kwargs)
        for hdf5_file, timestamps_file, depth_path, camera, dtype in zip(hdf5_files, timestamps_files, depth_paths, cameras, dtypes)
    ]
    subsets = [
        torch.utils.data.Subset(dataset, range(int(len(dataset) * range_[0]), int(len(dataset) * range_[2]), range_[1]))
        for dataset, range_ in zip(datasets, ranges)
    ]
    return torch.utils.data.ConcatDataset(subsets)


def get_dataloader_from_args(args: dict, logger: logging.Logger, shuffle: bool=True, train: bool=True):
    mode = "Training" if train else "Validation"
    config = args.train if train else args.val
    randomize_ctx = args.randomize_ctx if train else False
    logger.info(f"Loading {mode} dataset configs from {config['datasets']}...")

    datasets = []
    for dataset in config["datasets"]:
        with open(dataset, "r") as f:
            datasets.extend(yaml.safe_load(f)['datasets'])
    hdf5_files = [dataset['dataset_path'] for dataset in datasets]
    depth_paths = [dataset['depth_path'] for dataset in datasets]
    timestamps_files = [dataset.get('timestamps_path', None) for dataset in datasets]
    cameras = [dataset['camera'] for dataset in datasets]
    dtypes = [dataset['dtype'] for dataset in datasets]
    ranges = [dataset['range'] for dataset in datasets]

    kwargs = {
        "hdf5_files": hdf5_files, "timestamps_files": timestamps_files, "depth_paths": depth_paths,
        "min_numevents_ctx": args.min_numevents_ctx, "max_numevents_ctx": args.max_numevents_ctx,
        "time_ctx": args.time_ctx, "bucket": args.bucket, "w": args.frame_sizes[0], "h": args.frame_sizes[1],
        "randomize_ctx": randomize_ctx, "cameras": cameras, "ranges": ranges, "dtypes": dtypes
    }
    logger.info(f"Creating {mode} Dataloaders...")
    dataset = get_dataset_from_h5files(**kwargs)

    persistent_workers = True if train else False
    loader = DataLoader(
        dataset, batch_size=config["mini_batch"], pin_memory=True, persistent_workers=persistent_workers,
        shuffle=shuffle, num_workers=config["num_workers"], prefetch_factor=2*config["batch"]//config["mini_batch"],
        collate_fn=lambda batch: collate_fn_general(batch, [0], [1, 2, 3])
    )
    logger.info(f"{mode} Dataloader created with {len(dataset)} samples!")
    return loader


def get_dataloaders_from_args(args: dict, logger: logging.Logger, shuffle: bool=True):
    train_loader = get_dataloader_from_args(args, logger, shuffle, train=True)
    val_loader = get_dataloader_from_args(args, logger, False, train=False)
    return train_loader, val_loader
