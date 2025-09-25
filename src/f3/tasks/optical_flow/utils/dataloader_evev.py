import yaml
import logging
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from f3.utils import EventDatasetSingleHDF5, collate_fn_general


#! Master Dataloader
class TimeAlignedEventsAndEvents(EventDatasetSingleHDF5):
    """
        Creates a dataset of events. Returns a tuple of eventBlocks
        in the time ranges [-time_ctx, 0] and [0, timectx]
    """
    def __init__(self, hdf5_file: str, timestamps_50khz_file: str,
                 min_numevents_ctx: int=200000, max_numevents_ctx: int=800000,
                 time_ctx: int=20000, time_pred: int=20000, time_flow: int=25000,
                 bucket: int=1000, w: int=1280, h: int=720, randomize_ctx: bool=True,
                 camera: str="left", dtype: str="m3ed"):
        """
        |-time_ctx - time_pred//2  ... -time_pred//2| ... |-time_ctx//2 ... t0 ... time_ctx//2| ... |time_flow - time_ctx - time_pred//2 ... time_flow - time_pred//2| ... time_flow

        Args:
            time_pred: the time length which the model predicts into the future (Can be negative for baselines, that is for event frames/voxelgrids, we can have time_pred=-time_ctx
            so that we can load -time_ctx//2 to time_ctx//2 events for the context)
            time_flow: time difference over which the flow is to be computed
        """
        if dtype == "m3ed":
            self.src_ofst_res = torch.tensor([(h - 720) // 2, (w - 1280) // 2, 720, 1280], dtype=torch.int32)
        elif dtype == "dsec":
            #! For DSEC we want to drop the last 40 rows of pixels
            #! since there is a shadow of the car in the bottom 40 rows
            self.src_ofst_res = torch.tensor([(h - 480) // 2, (w - 640) // 2, 440, 640], dtype=torch.int32)
        elif dtype == "mvsec":
            self.src_ofst_res = torch.tensor([(h - 260) // 2, (w - 346) // 2, 260, 346], dtype=torch.int32)
        else:
            raise ValueError("Invalid dataset type!")

        self.time_flow = time_flow
        self.time_pred_half = time_pred // 2

        super(TimeAlignedEventsAndEvents, self).__init__(
            hdf5_file=hdf5_file, timestamps_50khz_file=timestamps_50khz_file, w=w, h=h,
            min_numevents_ctx=min_numevents_ctx, max_numevents_ctx=max_numevents_ctx,
            time_ctx=time_ctx, time_pred=time_pred, bucket=bucket, randomize_ctx=randomize_ctx,
            camera=camera, dtype=dtype
        )
        self.logger.info("### OPTICAL FLOW DATALOADER ###")

    def process_metadata(self):
        if self.load_metadata(keys=["camera", "min_numevents_ctx", "time_ctx", "time_pred", "time_flow"], fname="metadata_flow_evev.json"):
            self.logger.info(f"Dataset {Path(self.hdf5_fp).name} + {self.camera} loaded successfully!: {self.numblocks} valid blocks found!")
        else:
            # valid starting points for the feature field to the left and prediction events to the right
            self.valid_0_points = []
            start_index = self.time_ctx + self.time_pred_half + self.bucket # start index for 50khz timestamps
            end_index = int(self.events_t[-1]) + self.time_pred_half - self.time_flow - self.bucket # end index for 50khz timestamps
            for t0 in tqdm(range(start_index, end_index, self.time_flow), desc="Metadata Optical Flow"):
                ev_t0 = self.timestamps_50khz[(t0 - self.time_pred_half) // 20]
                ev_t0_old = self.timestamps_50khz[(t0 - self.time_ctx - self.time_pred_half) // 20]
                ev_t1 = self.timestamps_50khz[(t0 + self.time_flow - self.time_pred_half) // 20]
                ev_t1_old = self.timestamps_50khz[(t0 + self.time_flow - self.time_ctx - self.time_pred_half) // 20]

                pastdiff = int(ev_t0 - ev_t0_old - 1)
                futudiff = int(ev_t1 - ev_t1_old - 1)
                if pastdiff >= self.min_numevents_ctx and futudiff >= self.min_numevents_ctx:
                    self.valid_0_points.append(t0) # time in us
                    self.logger.info(f"Valid point: {t0}us")
            self.numblocks = len(self.valid_0_points) # number of data points we have for training and testing
            self.logger.info(f"Dataset {Path(self.hdf5_fp).name} loaded successfully!: {self.numblocks} valid blocks found!")
            self.metadata = {
                "camera": self.camera,
                "min_numevents_ctx": self.min_numevents_ctx,
                "time_ctx": self.time_ctx,
                "time_pred": self.time_pred,
                "time_flow": self.time_flow,
                "valid_0_points": self.valid_0_points
            }
            self.save_metadata("metadata_flow_evev.json")

    def __getitem__(self, idx):
        t0 = self.valid_0_points[idx]
        t0_new = t0 + self.time_flow

        ctx_old, totcnt_old = self.get_ctx_fixedtime(t0 - self.time_pred_half) # load from t0 - time_ctx - time_pred//2 to t0 - time_pred//2
        ctx_new, totcnt_new = self.get_ctx_fixedtime(t0_new - self.time_pred_half) # load from t0_new - time_ctx - time_pred//2 to t0_new - time_pred//2
        ctx_flow, totcnt_flow = self.get_ctx_fixedtime(t0 + self.time_ctx // 2) # load from t0 - time_ctx//2 to t0 + time_ctx//2 #! These are the events on which are visualized for the flow!

        return ctx_old, ctx_new, ctx_flow, totcnt_old, totcnt_new, totcnt_flow, self.src_ofst_res


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
        TimeAlignedEventsAndEvents(hdf5_file, timestamps_file, camera=camera, dtype=dtype, **kwargs)
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
        "time_ctx": args.time_ctx, "time_pred": args.time_pred, "time_flow": args.time_flow,
        "bucket": args.bucket,  "w": args.frame_sizes[0], "h": args.frame_sizes[1],
        "randomize_ctx": randomize_ctx, "cameras": cameras, "ranges": ranges, "dtypes": dtypes
    }
    logger.info(f"Creating {mode} Dataloaders...")
    dataset = get_dataset_from_h5files(**kwargs)

    persistent_workers = True if train else False
    loader = DataLoader(
        dataset, batch_size=config["mini_batch"], pin_memory=True, persistent_workers=persistent_workers,
        shuffle=shuffle, num_workers=config["num_workers"], prefetch_factor=2*config["batch"]//config["mini_batch"],
        collate_fn=lambda batch: collate_fn_general(batch, [0, 1, 2], [3, 4, 5, 6])
    )
    logger.info(f"{mode} Dataloader created with {len(dataset)} samples!")
    return loader


def get_dataloaders_from_args(args: dict, logger: logging.Logger, shuffle: bool=True):
    train_loader = get_dataloader_from_args(args, logger, shuffle, train=True)
    val_loader = get_dataloader_from_args(args, logger, False, train=False)
    return train_loader, val_loader
