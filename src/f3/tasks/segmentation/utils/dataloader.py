import h5py
import yaml
import logging
import hdf5plugin
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torchvision.io import decode_image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize, pad

from f3.utils import EventDatasetSingleHDF5, collate_fn_general
from f3.tasks.segmentation.utils import classes19_to_classes11


#! Master Dataloader
class TimeAlignedSegmentationAndEvents(EventDatasetSingleHDF5):
    """
        Loads the segmentation dataset and aligns the events with them.
    """
    def __init__(self, hdf5_file: str, timestamps_50khz_file: str, semantics_path: str, num_labels: int=11,
                 min_numevents_ctx: int=200000, max_numevents_ctx: int=800000, time_ctx: int=20000,
                 time_pred: int=20000, bucket: int=1000, w: int=1280, h: int=720, randomize_ctx: bool=True,
                 camera: str="left", dtype: str="m3ed"):
        """
            Args:
                semantics_path:
                    The h5 file containing semantic predictions and timestamps mapping to
                    the left prophesee camera FOR M3ED

                    The folder containing the semantic pngs and the timestamps file FOR DSEC
        """
        self.num_labels = num_labels
        if num_labels == 11:
            self.classes_mapping = classes19_to_classes11()

        # Load the semantics file
        if dtype == "m3ed":
            self.semantics = h5py.File(semantics_path, "r")
            self.ts_map = self.semantics['ts'][:]
            self.src_ofst_res = torch.tensor([0, 0, 720, 1280], dtype=torch.int32)
        elif dtype == "dsec":
            folder = Path(semantics_path)
            png_paths = sorted(
                (folder / "segmentation" / f"{num_labels}classes").glob("*.png"),
                key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or 0)
            )
            self.semantics_paths = [str(file) for file in png_paths]
            timestamps_path = folder / "segmentation" / f"{folder.stem}_semantic_timestamps.txt"
            self.ts_map = np.loadtxt(timestamps_path, dtype=np.uint64) # These are globally offset by a constant. We fix it in process_metadata
            png_paths = sorted(
                (folder / "images" / camera / "rectified").glob("*.png"),
                key=lambda x: int(''.join(filter(str.isdigit, x.stem)) or 0)
            )
            self.images_paths = [str(file) for file in png_paths]
            self.src_ofst_res = torch.tensor([(h - 480) // 2, (w - 640) // 2, 480, 640], dtype=torch.int32)
        else:
            raise ValueError("dtype should be either m3ed or dsec!")

        super(TimeAlignedSegmentationAndEvents, self).__init__(
            hdf5_file, timestamps_50khz_file, w, h, min_numevents_ctx, max_numevents_ctx,
            time_ctx, time_pred, bucket, randomize_ctx, camera, dtype
        )
        self.logger.info("### SEMANTIC DATALOADER ###")

    def process_metadata(self):
        if self.dtype == "dsec":
            self.ts_map -= self.hdf5_file['t_offset'][()].astype(np.uint64)
        if self.load_metadata(fname="metadata_semantics.json"):
            self.logger.info(f"Dataset {Path(self.hdf5_fp).name} + {self.camera} loaded successfully!: {self.numblocks} valid blocks found!")
        else:
            self.valid_0_points = []
            for idx in tqdm(range(self.ts_map.size), "Metadata Semantics"):
                t0 = self.ts_map[idx]
                t0 = t0 if t0 % 20 == 0 else (t0 // 20 + 1) * 20 # get it to be a multiple of 20
                if t0 < self.time_ctx: continue
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
                "time_pred": self.time_pred,
                "valid_0_points": self.valid_0_points
            }
            self.save_metadata(fname="metadata_semantics.json")

    def __getitem__(self, idx):
        fidx = self.valid_0_points[idx]
        t0 = self.ts_map[fidx] # time in us
        ctx, totcnt = self.get_ctx_fixedtime(t0)
        if self.dtype == "dsec":
            ovc = decode_image(self.images_paths[fidx])
            semantics = decode_image(self.semantics_paths[fidx]).squeeze() # 1, 440, 640 -> 440, 640
            ovc = resize(ovc, (self.trgt_res[1], self.trgt_res[0])).permute(1, 2, 0) # 720, 1280, 3 
            l = r = (self.trgt_res[0] - 640)//2
            t, b = (self.trgt_res[1] - 480)//2, (self.trgt_res[1] - 400)//2
            semantics = pad(semantics, (l, t, r, b), fill=255) # 480, 640 -> 720, 1280
        else:
            ovc = self.semantics['warped_image'][fidx] # (720, 1280, 1) or (720, 1280, 3)
            if ovc.shape[-1] == 1: ovc = np.repeat(ovc, 3, axis=-1)
            semantics = self.semantics['predictions'][fidx].squeeze() # 720, 1280, 1 -> 720, 1280
            if self.num_labels == 11:
                semantics = self.classes_mapping[semantics]
            ovc = torch.tensor(ovc, dtype=torch.uint8)
            semantics = torch.tensor(semantics, dtype=torch.uint8)
        return ctx, totcnt, ovc, semantics, self.src_ofst_res


def get_dataset_from_h5files(hdf5_files: list[str], timestamps_files: list[str], semantics_paths: list[str],
                             cameras: list[str]=None, dtypes: list[str]=None, ranges: list[list[float, int, float]]=None, **kwargs):
    assert len(hdf5_files) == len(timestamps_files), "The number of hdf5 files and timestamps files should be the same!"
    if cameras is None:
        cameras = ["left"] * len(hdf5_files)   # default to left camera
    if ranges is None:
        ranges = [[0, 1, 1]] * len(hdf5_files) # default to the whole dataset start, step, stop -> start and stop are fractions
    if dtypes is None:
        dtypes = ["m3ed"] * len(hdf5_files)    # default to m3ed dataset
    
    datasets = [
        TimeAlignedSegmentationAndEvents(hdf5_file, timestamps_file, semantics_path, camera=camera, dtype=dtype, **kwargs)
        for hdf5_file, timestamps_file, semantics_path, camera, dtype in zip(hdf5_files, timestamps_files, semantics_paths, cameras, dtypes)
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
    semantics_paths = [dataset['semantics_path'] for dataset in datasets]
    timestamps_files = [dataset['timestamps_path'] for dataset in datasets]
    cameras = [dataset['camera'] for dataset in datasets]
    dtypes = [dataset['dtype'] for dataset in datasets]
    ranges = [dataset['range'] for dataset in datasets]

    kwargs = {
        "hdf5_files": hdf5_files, "timestamps_files": timestamps_files, "semantics_paths": semantics_paths,
        "min_numevents_ctx": args.min_numevents_ctx, "max_numevents_ctx": args.max_numevents_ctx,
        "time_ctx": args.time_ctx, "time_pred": args.time_pred, "bucket": args.bucket, "num_labels": args.num_labels,
        "w": args.frame_sizes[0], "h": args.frame_sizes[1], "randomize_ctx": randomize_ctx,
        "cameras": cameras, "ranges": ranges, "dtypes": dtypes
    }
    logger.info(f"Creating {mode} Dataloaders...")
    dataset = get_dataset_from_h5files(**kwargs)

    persistent_workers = True if train else False
    loader = DataLoader(
        dataset, batch_size=config["mini_batch"], pin_memory=True, persistent_workers=persistent_workers,
        shuffle=shuffle, num_workers=config["num_workers"], prefetch_factor=2*config["batch"]//config["mini_batch"],
        collate_fn=lambda batch: collate_fn_general(batch, [0], [1, 2, 3, 4])
    )
    logger.info(f"{mode} Dataloader created with {len(dataset)} samples!")
    return loader


def get_dataloaders_from_args(args: dict, logger: logging.Logger, shuffle: bool=True):
    train_loader = get_dataloader_from_args(args, logger, shuffle, train=True)
    val_loader = get_dataloader_from_args(args, logger, False, train=False)
    return train_loader, val_loader
