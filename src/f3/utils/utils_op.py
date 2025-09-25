import torch
import torch.nn.functional as F


def dyadic_embedding(x: torch.Tensor, E: int, include_self: bool=True)->torch.Tensor:
    # applies position embedding emb_2i,j = sin(2^i * x_j) emb_2i+1,j=cos(2^i * x_j)
    # x: B x N x d
    # output: B x N x L
    dyadic_frequencies = 2**torch.arange(E, device=x.device) * torch.pi # E
    sin = torch.sin(x.unsqueeze(-1) * dyadic_frequencies.unsqueeze(0)) # B x N x d x E
    cos = torch.cos(x.unsqueeze(-1) * dyadic_frequencies.unsqueeze(0)) # B x N x d x E
    embedding = torch.cat([sin, cos], dim=-1) # B x N x d x 2E
    if include_self:
        embedding = torch.cat([embedding, x.unsqueeze(-1)], dim=-1) # B x N x d x 2E + 1
    return embedding


def broadcast_CELoss(pred: torch.Tensor, target: torch.Tensor, reduction: str="mean")->torch.Tensor:
    # pred: B x N x 1 or B x N x 2 probabilities
    # target: B x N x 1 or B x N x 2 target {0,1}
    # reduction: str, mean or sum
    return F.binary_cross_entropy(pred, target, reduction=reduction)


def broadcast_MSELoss(pred: torch.Tensor, target: torch.Tensor, reduction: str="mean")->torch.Tensor:
    # pred: B x N x 1 or B x N x 2 probabilities
    # target: B x N x 1 or B x N x 2 target {0,1}
    # reduction: str, mean or sum
    return F.mse_loss(pred, target, reduction=reduction)


def squashed_CELoss(pred: torch.Tensor, targets: torch.Tensor, reduction: str="mean", scale: float=1.0)->torch.Tensor:
    # pred: B x W x H or B x W x H x 2 # [0,1] contains probabilities for 2 non exclusive classes
    # targets: B x W x H or B x W x H x 2
    # reduction: str, mean or sum
    pos = targets.sum(dim=(0,1,2), keepdim=True)
    tot = targets.shape[0] * targets.shape[1] * targets.shape[2]
    pos_weight = ((tot - pos) / pos).expand_as(targets)
    return F.binary_cross_entropy(pred, targets, reduction=reduction, weight=pos_weight) * scale


def squashed_CELossLogits(pred: torch.Tensor, targets: torch.Tensor, reduction: str="mean", scale: float=1.0)->torch.Tensor:
    # pred: B x W x H x 1 or B x W x H x 2 # logits
    # targets: B x W x H x 1 or B x W x H x 2
    # reduction: str, mean or sum
    pos = targets.sum(dim=(1,2), keepdim=True)
    tot = targets.shape[1] * targets.shape[2]
    pos_weight = ((tot - pos) / pos).expand_as(targets)
    return F.binary_cross_entropy_with_logits(pred, targets, reduction=reduction, pos_weight=pos_weight) * scale


def squashed_MSELoss(pred: torch.Tensor, targets: torch.Tensor, reduction: str="mean", scale: float=1.0)->torch.Tensor:
    # pred: B x W x H or B x W x H x 2 # [0,1] contains probabilities for 2 non exclusive classes
    # targets: B x W x H or B x W x H x 2
    # reduction: str, mean or sum
    return F.mse_loss(pred, targets, reduction=reduction) * scale


class VoxelBlurLoss(torch.nn.Module):
    """
    VoxelMSELoss: Voxel Mean Squared Error Loss

    type: str, type of loss, mse, blurredtime, blurred3d
    """
    def __init__(self, type: str="mse", scale: float=1.0, kernel_size: int=5,
                 sigma: float=1.0, gamma: float=2.0):
        super(VoxelBlurLoss, self).__init__()
        self.type = type
        self.sigma = sigma
        self.scale = scale
        self.gamma = gamma
        self.kernel_size = kernel_size

        if "blurred" in self.type:
            self.kernel = self.create_gaussian_kernel_1d(self.kernel_size, self.sigma).to("cuda")

        FUNC_MAP = {
            "bc": self.voxel_BCLoss,
            "ce": self.voxel_CELoss,
            "mse": self.voxel_MSELoss,
            "bcblurredtime": self.voxel_blurred_time_BCLoss,
            "ceblurredtime": self.voxel_blurred_time_CELoss,
            "mseblurred3d": self.voxel_blurred_MSELoss,
            "mseblurredtime": self.voxel_blurred_time_MSELoss,
            "focalblurredtime": self.voxel_blurred_time_FocalLoss
        }
        self.forward = FUNC_MAP[self.type]

    @staticmethod
    def create_gaussian_kernel_1d(size: int, sigma: float)->torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size//2
        kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel

    def apply_gaussian_blur_1d(self, tensor: torch.Tensor, kernel: torch.Tensor, dim: int)->torch.Tensor:
        if dim == 2:
            tensor = tensor.permute(0, 3, 2, 1).contiguous()
        elif dim == 3:
            tensor = tensor.permute(0, 1, 3, 2).contiguous()

        B, D1, D2, D3 = tensor.shape
        tensor = tensor.view(B*D1*D2, 1, D3)
        kernel = kernel.view(1, 1, -1)

        blurred_tensor = F.conv1d(tensor, kernel, padding='same')
        blurred_tensor = blurred_tensor.view(B, D1, D2, D3)

        if dim == 2:
            blurred_tensor = blurred_tensor.permute(0, 3, 2, 1).contiguous()
        elif dim == 3:
            blurred_tensor = blurred_tensor.permute(0, 1, 3, 2).contiguous()
        return blurred_tensor

    def blurtime(self, tensor: torch.Tensor)->torch.Tensor:
        blurred_gt = tensor.clone()
        B, H, W, T = blurred_gt.shape
        blurred_gt = blurred_gt.view(B*H*W, 1, T)
        blurred_gt = F.conv1d(blurred_gt, self.kernel.view(1, 1, -1), padding='same')
        blurred_gt = blurred_gt.view(B, H, W, T)
        blurred_gt /= blurred_gt.max()
        return blurred_gt

    def reduce(self, loss: torch.Tensor, reduction: str)->torch.Tensor:
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise ValueError("Invalid reduction type")

    def voxel_MSELoss(self, pred: torch.Tensor, targets: torch.Tensor, reduction: str="mean")->torch.Tensor:
        # pred: B x H x W x T or B x H x W x T x 2 # [0,1] contains logits for 2 non exclusive classes
        # gt: B x H x W x T or B x H x W x T x 2 # {0,1}
        # reduction: str, mean or sum
        return F.mse_loss(torch.sigmoid(pred), targets, reduction=reduction) * self.scale

    def voxel_blurred_MSELoss(self, pred: torch.Tensor, gt: torch.Tensor, reduction: str="mean")->torch.Tensor:
        # pred: B x H x W x T or B x H x W x T x 2 # [0,1] contains logits for 2 non exclusive classes
        # gt: B x H x W x T or B x H x W x T x 2 # {0,1}
        # reduction: str, mean or sum
        if len(gt.shape) == 4:
            blurred_gt = gt.clone()
            for dim in [1, 2, 3]:
                blurred_gt = self.apply_gaussian_blur_1d(blurred_gt.float(), self.kernel, dim)
            blurred_gt /= blurred_gt.max()
        elif len(gt.shape) == 5 and gt.shape[-1] == 2:
            blurred_gt = torch.empty_like(gt)
            for i in range(2):
                blurred_gt[..., i] = gt[..., i].clone()
                for dim in [1, 2, 3]:
                    blurred_gt[..., i] = self.apply_gaussian_blur_1d(blurred_gt[..., i].float(), self.kernel, dim)
                blurred_gt[..., i] /= blurred_gt[..., i].max()
        else:
            raise ValueError("Invalid shape for gt must be B x H x W x T or B x H x W x T x 2")
        return F.mse_loss(torch.sigmoid(pred), blurred_gt, reduction=reduction) * self.scale

    def voxel_blurred_time_MSELoss(self, pred: torch.Tensor, gt: torch.Tensor, reduction: str="mean")->torch.Tensor:
        # pred: B x H x W x T or B x H x W x T x 2 # [0,1] contains logits for 2 non exclusive classes
        # gt: B x H x W x T or B x H x W x T x 2 # {0,1}
        # reduction: str, mean or sum
        blurred_gt = self.blurtime(gt)
        return F.mse_loss(torch.sigmoid(pred), blurred_gt, reduction=reduction) * self.scale

    def voxel_BCLoss(self, pred: torch.Tensor, gt: torch.Tensor, reduction: str="mean")->torch.Tensor:
        # pred: B x H x W x T or B x H x W x T x 2 # logits
        # gt: B x H x W x T or B x H x W x T x 2 # {0,1}
        # reduction: str, mean or sum
        pred = torch.sigmoid(pred)
        bc = -torch.log(torch.sqrt(pred * gt) + torch.sqrt((1 - pred) * (1 - gt)))
        return self.reduce(bc, reduction=reduction) * self.scale
    
    def voxel_blurred_time_BCLoss(self, pred: torch.Tensor, gt: torch.Tensor, reduction: str="mean")->torch.Tensor:
        # pred: B x H x W x T or B x H x W x T x 2 # logits
        # gt: B x H x W x T or B x H x W x T x 2 # {0,1}
        # reduction: str, mean or sum
        return self.voxel_BCLoss(pred, self.blurtime(gt), reduction=reduction)

    def voxel_CELoss(self, pred: torch.Tensor, gt: torch.Tensor, reduction: str="mean")->torch.Tensor:
        # pred: B x H x W x T x 1 or B x H x W x T x 2 # logits
        # gt: B x H x W x T x 1 or B x H x W x T x 2 # {0,1}
        # reduction: str, mean or sum
        pos = gt.sum(dim=(1,2,3), keepdim=True)
        tot = gt.shape[1] * gt.shape[2] * gt.shape[3]
        pos_weight = ((tot - pos) / pos).expand_as(gt)
        return F.binary_cross_entropy_with_logits(pred, gt, reduction=reduction, pos_weight=pos_weight) * self.scale

    def voxel_blurred_time_CELoss(self, pred: torch.Tensor, gt: torch.Tensor, reduction: str="mean")->torch.Tensor:
        # pred: B x H x W x T x 1 or B x H x W x T x 2 # logits
        # gt: B x H x W x T x 1 or B x H x W x T x 2 # {0,1}
        # reduction: str, mean or sum
        pos = gt.sum(dim=(1,2,3), keepdim=True)
        tot = gt.shape[1] * gt.shape[2] * gt.shape[3]
        pos_weight = ((tot - pos) / pos).expand_as(gt)
        return F.binary_cross_entropy_with_logits(pred, self.blurtime(gt), reduction=reduction, pos_weight=pos_weight) * self.scale

    def voxel_blurred_time_FocalLoss(self, pred: torch.Tensor, gt: torch.Tensor, reduction: str="mean")->torch.Tensor:
        # pred: B x H x W x T x 1 or B x H x W x T x 2 # logits
        # gt: B x H x W x T x 1 or B x H x W x T x 2 # {0,1}
        # reduction: str, mean or sum
        return voxel_FocalLoss(pred, self.blurtime(gt), reduction=reduction, scale=self.scale, gamma=self.gamma)


def voxel_FocalLoss(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor,
                    reduction: str="mean", scale: float=1.0, gamma: float=2.0)->torch.Tensor:
    # pred: B x H x W x T x 1 or B x H x W x T x 2 # logits
    # gt: B x H x W x T x 1 or B x H x W x T x 2 # {0,1}
    # valid_mask: B x W x H # {0,1}
    # reduction: str, mean or sum
    p = torch.sigmoid(pred)
    p_t = p * gt + (1 - p) * (1 - gt)
    valid_mask = valid_mask.unsqueeze(-1).expand_as(gt)

    ce_loss = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
    loss = ce_loss * ((1 - p_t) ** gamma)

    alpha = (1 - gt.sum(dim=(1,2,3), keepdim=True) / valid_mask.sum(dim=(1,2,3), keepdim=True)).expand_as(gt)
    loss *= alpha * gt + (1 - alpha) * (1 - gt)

    if reduction == "mean":
        return (loss * valid_mask).mean() * scale
    elif reduction == "sum":
        return (loss * valid_mask).sum() * scale
    elif reduction == "none":
        return loss * valid_mask


def voxel_EquiWeightedMSELoss(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, scale: float=1.0)->torch.Tensor:
    # pred: B x H x W x T x 1 or B x H x W x T x 2 # logits
    # gt: B x H x W x T x 1 or B x H x W x T x 2 # {0,1}
    # valid_mask: B x W x H # {0,1}
    p = torch.sigmoid(pred)
    valid_mask = valid_mask.unsqueeze(-1).expand_as(gt)

    mse_loss = F.mse_loss(p, gt, reduction="none")

    masks = (gt * valid_mask, (1 - gt) * valid_mask)
    counts = (masks[0].sum(), masks[1].sum())

    loss = 0.0
    if counts[0] > 0: loss += (mse_loss * masks[0]).sum() / counts[0]
    if counts[1] > 0: loss += (mse_loss * masks[1]).sum() / counts[1]
    return 0.5 * scale * loss


def voxel_magfft1d_MSELoss(pred: torch.Tensor, gt: torch.Tensor, reduction: str="mean",
                           num_components: int=10, scale: float=1.0)->torch.Tensor:
    # pred: B x H x W x T or B x H x W x T x 2 # [0,1] contains logits for 2 non exclusive classes
    # gt: B x H x W x T or B x H x W x T x 2 # {0,1}
    # reduction: str, mean or sum
    # num_components: int, number of frequency components to consider
    pred = torch.sigmoid(pred)
    if len(gt.shape) == 4: dim = -1
    elif len(gt.shape) == 5: dim = -2
    fft_pred = torch.abs(torch.fft.rfft(pred, dim=dim)[..., :num_components])
    fft_gt = torch.abs(torch.fft.rfft(gt.float(), dim=dim)[..., :num_components])
    return F.mse_loss(fft_pred, fft_gt, reduction=reduction) * scale
