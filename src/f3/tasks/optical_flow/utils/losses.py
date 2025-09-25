import torch
import torch.nn.functional as F


def warp_images_with_flow(image, flow):
    """
    Warp an image using optical flow while preserving gradients, including for flow.

    Args:
        image (torch.Tensor): Input image of shape (B, C, H, W).
        flow (torch.Tensor): Optical flow of shape (B, 2, H, W), with requires_grad=True.

    Returns:
        torch.Tensor: Warped image of shape (B, C, H, W).
    """
    B, C, H, W = image.shape
    y, x = torch.meshgrid(
        torch.arange(H, device=image.device),
        torch.arange(W, device=image.device),
        indexing='ij'
    )
    grid = torch.stack((x, y), dim=0).float()  # Shape: (2, H, W)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # Shape: (B, 2, H, W)

    warped_grid = grid + flow  # Shape: (B, 2, H, W)

    # Normalize grid to [-1, 1]
    warped_grid[:, 0, :, :] = 2.0 * (warped_grid[:, 0, :, :] / (W - 1)) - 1.0
    warped_grid[:, 1, :, :] = 2.0 * (warped_grid[:, 1, :, :] / (H - 1)) - 1.0

    warped_grid = warped_grid.permute(0, 2, 3, 1)
    warped_image = F.grid_sample(image, warped_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped_image


def charbonnier_loss(delta, valid_mask=None, alpha=0.45, epsilon=1e-3):
    if valid_mask is None:
        valid_mask = torch.ones_like(delta)
    loss = torch.mean(torch.pow(
        torch.square(delta) + torch.square(torch.tensor(epsilon)), alpha
    ) * valid_mask)
    return loss


@torch.compile()
def compute_smoothness_loss(flow, pyr_levels=4):
    total_smoothness_loss = 0.
    for i in range(pyr_levels):
        flow_pyr = F.interpolate(flow, scale_factor=1 / (2 ** i), mode='bilinear', align_corners=True) * (1 / (2 ** i))
        flow_ucrop = flow_pyr[:, :, 1:, :]
        flow_dcrop = flow_pyr[:, :, :-1, :]
        flow_lcrop = flow_pyr[:, :, :, 1:]
        flow_rcrop = flow_pyr[:, :, :, :-1]

        flow_ulcrop = flow_pyr[:, :, 1:, 1:]
        flow_drcrop = flow_pyr[:, :, :-1, :-1]
        flow_dlcrop = flow_pyr[:, :, :-1, 1:]
        flow_urcrop = flow_pyr[:, :, 1:, :-1]

        smoothness_loss = charbonnier_loss(flow_lcrop - flow_rcrop) + \
                          charbonnier_loss(flow_ucrop - flow_dcrop) + \
                          charbonnier_loss(flow_ulcrop - flow_drcrop) + \
                          charbonnier_loss(flow_dlcrop - flow_urcrop)
        total_smoothness_loss += smoothness_loss / 4
    return total_smoothness_loss / pyr_levels


def gaussian_kernel1d(size, sigma):
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def gaussian_kernel2d(size, sigma):
    kernel_1d = gaussian_kernel1d(size, sigma)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d


def gaussian_blur(image, kernel):
    """
    Apply Gaussian blur to an image.

    Args:
        image (torch.Tensor): Input image (B, C, H, W).
        kernel (torch.Tensor): Gaussian kernel for blurring.

    Returns:
        torch.Tensor: Blurred image.
    """
    kernel = kernel[None, None, :, :].expand(image.shape[1], -1, -1, -1)
    padding = (kernel.shape[-1] - 1) // 2
    blurred_image = F.conv2d(image, kernel, padding=padding, groups=image.shape[1])
    return blurred_image


def generate_image_pyramids_with_blur(image, num_levels, kernel):
    """
    Generate an image pyramid with Gaussian blur for smoother transitions.

    Args:
        image (torch.Tensor): Input image of shape (B, C, H, W).
        num_levels (int): Number of pyramid levels to generate.
        kernel (torch.Tensor): Gaussian kernel for blurring.

    Returns:
        list[torch.Tensor]: List of images at each pyramid level.
    """
    pyramids = [image]
    for i in range(1, num_levels):
        blurred_image = gaussian_blur(pyramids[-1], kernel)
        downsampled_image = F.interpolate(
            blurred_image,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=True
        )
        pyramids.append(downsampled_image)
    return pyramids


@torch.compile()
def compute_photometric_loss(prev_images, next_images, flow,
                             valid_mask=None, pyr_levels=4, kernel=None, normalize=True):
    """
    Compute photometric loss between two images warped with flow

    Args:
        prev_images: Previous images (B, C, H, W)
        next_images: Next images (B, C, H, W)
        flow: Optical flow (B, 2, H, W)
        valid_mask: Valid mask (B, 1, H, W)
        pyr_levels: Number of pyramid levels 4 -> 1, 1/2, 1/4, 1/8
    """
    B, C = prev_images.shape[:2]
    total_photometric_loss = 0.

    prev_images_pyrs = generate_image_pyramids_with_blur(prev_images, pyr_levels, kernel)
    next_images_pyrs = generate_image_pyramids_with_blur(next_images, pyr_levels, kernel)

    for i in range(pyr_levels):
        flow_pyr = F.interpolate(flow, scale_factor=1 / (2 ** i), mode='bilinear', align_corners=True) * (1 / (2 ** i))

        valid_mask_pyr = None
        if valid_mask is not None:
            valid_mask_pyr = F.interpolate(valid_mask, scale_factor=1 / (2 ** i), mode='bilinear', align_corners=True)

        next_images_warped = warp_images_with_flow(next_images_pyrs[i], flow_pyr)

        if normalize:
            normfactor = torch.cat([
                next_images_warped.reshape(B, C, -1), prev_images_pyrs[i].reshape(B, C, -1)
            ], dim=-1).norm(dim=-1)[:, :, None, None] + 1e-7
            next_images_warped = next_images_warped / normfactor
            prev_images_pyrs[i] = prev_images_pyrs[i] / normfactor

        photometric_loss = charbonnier_loss(next_images_warped - prev_images_pyrs[i], valid_mask_pyr)
        total_photometric_loss += photometric_loss
    return total_photometric_loss / pyr_levels
