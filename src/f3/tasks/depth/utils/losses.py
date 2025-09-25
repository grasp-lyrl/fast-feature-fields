import torch
from torch import nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.name = 'SiLogLoss'
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask] + 1e-6)
        return torch.sqrt(torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2))


class SiLogGradLoss(SiLogLoss):
    def __init__(self, lambd=0.5, alpha=0.5, scales=4):
        super().__init__(lambd)
        self.name = 'SiLogGradLoss'
        self.__alpha = alpha
        self.__scales = scales

    def forward(self, pred, target_gt, grad_gt, target_valid_mask, grad_valid_mask=None):
        """
        Use this loss to finetune a model on metric disparity from a LIDAR. This uses
        the gradient loss from non metric depth to guide the edges of the predicted disparity.

            pred: Prediction of disparity from the network
            target_gt: In this context the target disparity is from a LIDAR
                       and hence sparse, this is perhaps metric and to scale!
            grad_gt: This is non-metric depth from our trained model. This is dense, but not to scale.
            valid_mask: Mask to ignore invalid pixels in the target disparity.
        """
        l_silog = super().forward(pred, target_gt, target_valid_mask)

        # For gradient we need to scale and shift the disparity
        if grad_valid_mask is None:
            grad_valid_mask = torch.ones_like(grad_gt, dtype=torch.bool)

        pred_hat, grad_gt_hat = compute_scale_and_shift_mae(pred, grad_gt, grad_valid_mask)
        l_reg = compute_gradient_loss(pred_hat, grad_gt_hat, grad_valid_mask, scales=self.__scales)

        loss = l_silog + self.__alpha * l_reg
        return loss


def gradient_loss(prediction, target, mask):
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    batch_loss = torch.sum(image_loss) / torch.sum(M)
    return batch_loss


def compute_gradient_loss(prediction, target, mask, scales=4):
    """
        eqn 11 from https://arxiv.org/pdf/1907.01341    
        
        prediction and target should be the scaled and shifted disparity
    """
    total = 0
    for scale in range(scales):
        step = 2 ** scale
        total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step], mask[:, ::step, ::step])
    return total


# MAE version
def compute_scale_and_shift_mae(prediction, target, mask):
    """
        prediction and targets are the disparity predictions (Inverse Depth). Be careful
        
        prediction: B, H, W
        target: B, H, W
        mask: B, H, W
        
        I follow eqn 5,6,7 from https://arxiv.org/pdf/1907.01341
        
        I will try the trim loss later
        
        d_p_hat = \hat{d} in paper
        d_t_hat = \hat{d}^* in paper
    """
    B, H, W = prediction.shape
    prediction_masked = prediction.masked_fill(~mask, torch.nan).view(B, -1)
    t_p = torch.nanmedian(prediction_masked, dim=-1, keepdim=True)[0]
    s_p = torch.nanmean(torch.abs(prediction_masked - t_p), dim=-1)[:, None, None]
    
    target_masked = target.masked_fill(~mask, torch.nan).view(B, -1)
    t_t = torch.nanmedian(target_masked, dim=-1, keepdim=True)[0]
    s_t = torch.nanmean(torch.abs(target_masked - t_t), dim=-1)[:, None, None]
    
    d_p_hat = (prediction - t_p[:, :, None]) / s_p
    d_t_hat = (target - t_t[:, :, None]) / s_t
    return d_p_hat, d_t_hat


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4):
        super().__init__()
        self.__alpha = alpha
        self.__scales = scales
        self.name = "SSIMAELoss"

    def forward(self, prediction, target, mask):
        d_p_hat, d_t_hat = compute_scale_and_shift_mae(prediction, target, mask)
        l_ssimae = nn.functional.l1_loss(d_p_hat[mask], d_t_hat[mask])
        l_reg = compute_gradient_loss(d_p_hat, d_t_hat, mask, scales=self.__scales)
        
        loss = l_ssimae + self.__alpha * l_reg
        return loss
