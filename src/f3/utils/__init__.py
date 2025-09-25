from .dataloader import get_dataloader_from_args, get_dataloaders_from_args, EventDatasetSingleHDF5, BaseExtractor
from .utils_gen import *
from .utils_op import *
from .utils_model import *
from .utils_viz import *
from .utils_train import *
from .utils_val import *
from .utils_setup import *
# from .utils_timers import *
from .utils_MRHE import MultiResolutionHashEncoder


LOSSES = {
    "CELoss": broadcast_CELoss,
    "MSELoss": broadcast_MSELoss,
    "SquashedMSELoss": squashed_MSELoss,
    "SquashedCELoss": squashed_CELoss,
    "SquashedCELossLogits": squashed_CELossLogits,
    "EquiWeightedMSELoss": voxel_EquiWeightedMSELoss,
    "VoxelFocalLoss": voxel_FocalLoss,
    "VoxelCELoss": VoxelBlurLoss("ce"),
    "VoxelBCLoss": VoxelBlurLoss("bc"),
    "VoxelMSELoss": VoxelBlurLoss("mse"),
    "VoxelBlurredTimeCELoss": VoxelBlurLoss("ceblurredtime"),
    "VoxelBlurredTimeBCLoss": VoxelBlurLoss("bcblurredtime"),
    "VoxelBlurred3DMSELoss": VoxelBlurLoss("mseblurred3d"),
    "VoxelBlurredTimeMSELoss": VoxelBlurLoss("mseblurredtime"),
    "VoxelBlurredTimeFocalLoss": VoxelBlurLoss("focalblurredtime"),
    "VoxelMagFFTLoss": voxel_magfft1d_MSELoss
}