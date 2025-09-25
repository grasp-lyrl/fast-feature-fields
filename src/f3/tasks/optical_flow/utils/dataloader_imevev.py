import cv2
import h5py
import yaml
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from f3.utils import EventDatasetSingleHDF5, collate_fn_general


#! Master Dataloader
class TimeAlignedImagesAndEventsEvents(EventDatasetSingleHDF5):
    """
        Creates a dataset of events. Returns a tuple of eventBlocks
        in the time ranges [-time_ctx, 0] and [0, timectx]
    """
    def __init__(self, data_path: str, timestamps_50khz_file: str,
                 min_numevents_ctx: int=200000, max_numevents_ctx: int=800000,
                 time_ctx: int=20000, bucket: int=1000, w: int=1280, h: int=720,
                 randomize_ctx: bool=True, camera: str="left", dtype: str="m3ed"):
        
        if dtype == "m3ed":
            eventh5_path = data_path
            self.ts_map = h5py.File(eventh5_path, 'r')['ovc/ts'][:]
            self.src_ofst_res = torch.tensor([(h - 720) // 2, (w - 1280) // 2, 720, 1280], dtype=torch.int32)
        elif dtype == "dsec":
            folder = Path(data_path)
            eventh5_path = folder / "events" / camera / "events.h5"
            image_path = folder / "images" / camera / "rectified"
            timestamps_path = folder / "images" / "timestamps.txt"
            png_paths = sorted(
                image_path.glob("*.png"),
                key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or 0)
            )
            self.image_paths = [str(path) for path in png_paths]
            self.ts_map = np.genfromtxt(timestamps_path, dtype=np.uint64) -\
                          h5py.File(eventh5_path, "r")['t_offset'][()].astype(np.uint64)
            self.src_ofst_res = torch.tensor([(h - 480) // 2, (w - 640) // 2, 480, 640], dtype=torch.int32)
        else:
            raise ValueError("Invalid dataset type!")
        
        super(TimeAlignedImagesAndEventsEvents, self).__init__(
            hdf5_file=eventh5_path, timestamps_50khz_file=timestamps_50khz_file, w=w, h=h,
            min_numevents_ctx=min_numevents_ctx, max_numevents_ctx=max_numevents_ctx,
            time_ctx=time_ctx, bucket=bucket, randomize_ctx=randomize_ctx, camera=camera, dtype=dtype
        )
        self.logger.info("### OPTICAL FLOW DATALOADER ###")

        if self.dtype == "m3ed":
            self.im_path = 'ovc/rgb/data' if 'ovc/rgb' in self.hdf5_file else 'ovc/left/data'

    def process_metadata(self):
        if self.load_metadata(keys=["camera", "min_numevents_ctx", "time_ctx"], fname="metadata_flow_imimevev.json"):
            self.logger.info(f"Dataset {Path(self.hdf5_fp).name} + {self.camera} loaded successfully!: {self.numblocks} valid blocks found!")
        else:
            # valid starting points for the feature field to the left and prediction events to the right
            self.valid_0_points = []
            for idx in tqdm(range(self.ts_map.size - 1), "Metadata Optical Flow"):
                t0, t1 = self.ts_map[idx], self.ts_map[idx + 1]
                t0 = t0 if t0 % 20 == 0 else (t0 // 20 + 1) * 20 # get it to be a multiple of 20
                t1 = t1 if t1 % 20 == 0 else (t1 // 20 + 1) * 20 # get it to be a multiple of 20
                if t0 < self.time_ctx: continue
                
                cnt0 = self.timestamps_50khz[t0 // 20] - self.timestamps_50khz[(t0 - self.time_ctx) // 20] - 1
                cnt1 = self.timestamps_50khz[t1 // 20] - self.timestamps_50khz[(t1 - self.time_ctx) // 20] - 1
                if cnt0 >= self.min_numevents_ctx and cnt1 >= self.min_numevents_ctx:
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
            self.save_metadata("metadata_flow_imimevev.json")

    def __len__(self):
        return self.numblocks

    def __getitem__(self, idx):
        fidx = self.valid_0_points[idx]
        t0 = self.ts_map[fidx]
        ctx0, totcnt0 = self.get_ctx_fixedtime(t0)

        t1 = self.ts_map[fidx + 1]
        ctx1, totcnt1 = self.get_ctx_fixedtime(t1)

        if self.dtype == "m3ed":
            img = self.hdf5_file[self.im_path][fidx]  # (720, 1280, 3)
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        elif self.dtype == "dsec":
            img = cv2.imread(self.image_paths[fidx])
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return ctx0, ctx1, totcnt0, totcnt1, img, self.src_ofst_res


def get_dataset_from_h5files(hdf5_files: list[str], timestamps_files: list[str], cameras: list[str]=None,
                             dtypes: list[str]=None, ranges: list[list[float, int, float]]=None, **kwargs):
    assert len(hdf5_files) == len(timestamps_files), "The number of hdf5 files and timestamps files should be the same!"
    if cameras is None:
        cameras = ["left"] * len(hdf5_files)   # default to left camera
    if ranges is None:
        ranges = [[0, 1, 1]] * len(hdf5_files) # default to the whole dataset start, step, stop -> start and stop are fractions
    if dtypes is None:
        dtypes = ["m3ed"] * len(hdf5_files)    # default to m3ed dataset
    
    datasets = [
        TimeAlignedImagesAndEventsEvents(hdf5_file, timestamps_file, camera=camera, dtype=dtype, **kwargs)
        for hdf5_file, timestamps_file, camera, dtype in zip(hdf5_files, timestamps_files, cameras, dtypes)
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
    cameras = [dataset['camera'] for dataset in datasets]
    dtypes = [dataset['dtype'] for dataset in datasets]
    ranges = [dataset['range'] for dataset in datasets]

    kwargs = {
        "hdf5_files": hdf5_files, "timestamps_files": timestamps_files,
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
        collate_fn=lambda batch: collate_fn_general(batch, [0, 1], [2, 3, 4, 5])
    )
    logger.info(f"{mode} Dataloader created with {len(dataset)} samples!")
    return loader


def get_dataloaders_from_args(args: dict, logger: logging.Logger, shuffle: bool=True):
    train_loader = get_dataloader_from_args(args, logger, shuffle, train=True)
    val_loader = get_dataloader_from_args(args, logger, False, train=False)
    return train_loader, val_loader
