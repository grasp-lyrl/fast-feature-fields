import yaml
import h5py
import imageio
import logging
import hdf5plugin
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from f3.utils import EventDatasetSingleHDF5, collate_fn_general


#! Master Dataloader for flow evaluation
class TimeAlignedGTFlowAndEvents(EventDatasetSingleHDF5):
    def __init__(self, hdf5_file: str, timestamps_50khz_file: str, flow_path: str,
                 min_numevents_ctx: int=200000, max_numevents_ctx: int=800000,
                 time_ctx: int=20000, bucket: int=1000, w: int=1280, h: int=720,
                 randomize_ctx: bool=True, camera: str="left", dtype: str="m3ed"):
        """
            Events: |-time_ctx//2 ... t0 ... time_ctx//2| 
            GT Flow at t0

            flow_path: Path to the GT Flow file (m3ed/mvsec) or the folder containing the GT Flow pngs (dsec)
        """
        if dtype == "m3ed":
            self.gtflow_h5 = h5py.File(flow_path, "r")
            self.gtflow_x = self.gtflow_h5['flow/prophesee/left/x'] # 32bit float (N, 720, 1280)
            self.gtflow_y = self.gtflow_h5['flow/prophesee/left/y'] # 32bit float (N, 720, 1280)
            self.ts_map = self.gtflow_h5['ts'][:]
            self.src_ofst_res = torch.tensor([(h - 720) // 2, (w - 1280) // 2, 720, 1280], dtype=torch.int32)

        elif dtype == "dsec":
            #! For DSEC we want to drop the last 40 rows of pixels
            #! since there is a shadow of the car in the bottom 40 rows
            folder = Path(flow_path)
            png_paths = sorted(
                (folder / "forward").glob("*.png"),
                key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or 0)
            )
            self.flow_paths = [str(path) for path in png_paths]
            timestamps_path = folder / "forward_timestamps.txt"
            self.ts_map = np.genfromtxt(timestamps_path, dtype=np.uint64, delimiter=',', skip_header=1)[:, 0] -\
                        h5py.File(hdf5_file, "r")['t_offset'][()].astype(np.uint64)
            self.src_ofst_res = torch.tensor([(h - 480) // 2, (w - 640) // 2, 450, 640], dtype=torch.int32)

        elif dtype == "mvsec":
            flow_path = Path(flow_path)
            if flow_path.suffix == ".hdf5":
                self.ish5 = True
                self.gtflow_h5 = h5py.File(flow_path, "r")
                self.gtflow = self.gtflow_h5['davis/left/flow_dist']
                self.ts_map = ((
                    self.gtflow_h5['davis/left/flow_dist_ts'][:] -\
                    h5py.File(hdf5_file, "r").attrs["absolute_start_time"]
                ) * 1e6).astype(np.uint64)
            elif flow_path.is_dir():
                self.ish5 = False
                gtflow_dir = flow_path / "optical_flow"
                timestamps_path = flow_path / "timestamps_flow.txt"
                self.flow_paths = [str(file) for file in sorted(
                    gtflow_dir.glob("*.npy"),
                    key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or 0)
                )]
                self.ts_map = ((
                    np.loadtxt(timestamps_path, dtype=np.float64) -\
                    h5py.File(hdf5_file, "r").attrs["absolute_start_time"]
                ) * 1e6).astype(np.uint64)
            else:
                raise ValueError(f"{flow_path} type not supported!!!")
            self.src_ofst_res = torch.tensor([(h - 260) // 2, (w - 346) // 2, 260, 346], dtype=torch.int32)

        else:
            raise ValueError("Invalid dataset type!")

        super(TimeAlignedGTFlowAndEvents, self).__init__(
            hdf5_file=hdf5_file, timestamps_50khz_file=timestamps_50khz_file, w=w, h=h,
            min_numevents_ctx=min_numevents_ctx, max_numevents_ctx=max_numevents_ctx,
            time_ctx=time_ctx, bucket=bucket, randomize_ctx=randomize_ctx,
            camera=camera, dtype=dtype
        )
        self.logger.info("### OPTICAL FLOW DATALOADER ###")

    def process_metadata(self):
        if self.load_metadata(keys=["camera", "min_numevents_ctx", "time_ctx"], fname="metadata_flow_gtev.json"):
            self.logger.info(f"Dataset {Path(self.hdf5_fp).name} + {self.camera} loaded successfully!: {self.numblocks} valid blocks found!")
        else:
            # valid starting points for the feature field to the left and prediction events to the right
            self.valid_0_points = []
            for idx in tqdm(range(self.ts_map.size), desc="Metadata Optical Flow"):
                t0 = self.ts_map[idx] + self.time_ctx // 2
                t0 = t0 if t0 % 20 == 0 else (t0 // 20 + 1) * 20 # get it to be a multiple of 20
                if t0 <= self.time_ctx: continue
                cnt = self.timestamps_50khz[t0 // 20] - self.timestamps_50khz[(t0 - self.time_ctx) // 20] - 1
                if cnt >= self.min_numevents_ctx:
                    self.valid_0_points.append(idx)
                    self.logger.info(f"Valid index: {idx}!")
            self.numblocks = len(self.valid_0_points) # number of data points we have for training and testing
            self.logger.info(f"Dataset {Path(self.hdf5_fp).name} loaded successfully!: {self.numblocks} valid blocks found!")
            self.metadata = {
                "camera": self.camera,
                "min_numevents_ctx": self.min_numevents_ctx,
                "time_ctx": self.time_ctx,
                "valid_0_points": self.valid_0_points
            }
            self.save_metadata("metadata_flow_gtev.json")

    def __getitem__(self, idx):
        fidx = self.valid_0_points[idx]
        t0 = self.ts_map[fidx] + self.time_ctx // 2
        t0 = t0 if t0 % 20 == 0 else (t0 // 20 + 1) * 20 # get it to be a multiple of 20
        ctx_flow, totcnt_flow = self.get_ctx_fixedtime(t0) # Load from [-time_ctx//2 ... t0 ... time_ctx//2]

        if self.dtype == "m3ed":
            flow_x = torch.from_numpy(self.gtflow_x[fidx]) # (720, 1280)
            flow_y = torch.from_numpy(self.gtflow_y[fidx]) # (720, 1280)
            flow = torch.stack([flow_x, flow_y], dim=0) # (2, 720, 1280)
        elif self.dtype == "dsec":
            flow = imageio.imread(self.flow_paths[fidx], format="PNG-FI")[..., :2] # (480, 640, 2)
            flow = torch.tensor(flow, dtype=torch.float32).permute(2, 0, 1) # (2, 480, 640)
            flow  = (flow - 2**15) / 128.0
        elif self.dtype == "mvsec":
            if self.ish5:
                flow = torch.from_numpy(self.gtflow[fidx]) # (2, 260, 346)
            else:
                flow = np.load(self.flow_paths[fidx]) # (2, 260, 346)

        return ctx_flow, totcnt_flow, flow, self.src_ofst_res


def get_dataset_from_h5files(hdf5_files: list[str], timestamps_files: list[str], flow_paths: list[str], cameras: list[str]=None,
                             dtypes: list[str]=None, ranges: list[list[float, int, float]]=None, **kwargs):
    assert len(hdf5_files) == len(timestamps_files) == len(flow_paths), "Number of datasets should be the same!"
    if cameras is None:
        cameras = ["left"] * len(hdf5_files)   # default to left camera
    if ranges is None:
        ranges = [[0, 1, 1]] * len(hdf5_files) # default to the whole dataset start, step, stop -> start and stop are fractions
    if dtypes is None:
        dtypes = ["m3ed"] * len(hdf5_files)    # default to m3ed dataset
    
    datasets = [
        TimeAlignedGTFlowAndEvents(hdf5_file, timestamps_file, flow_path, camera=camera, dtype=dtype, **kwargs)
        for hdf5_file, timestamps_file, flow_path, camera, dtype in zip(hdf5_files, timestamps_files, flow_paths, cameras, dtypes)
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
    timestamps_files = [dataset['timestamps_path'] for dataset in datasets]
    flow_paths = [dataset['flow_path'] for dataset in datasets]
    cameras = [dataset['camera'] for dataset in datasets]
    dtypes = [dataset['dtype'] for dataset in datasets]
    ranges = [dataset['range'] for dataset in datasets]

    kwargs = {
        "hdf5_files": hdf5_files, "timestamps_files": timestamps_files, "flow_paths": flow_paths,
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
