import cv2
import torch
import numpy as np
from sklearn.decomposition import PCA

from f3 import init_event_model, load_weights_ckpt


class StereoEventFFMatcher:
    def __init__(
        self,
        eventff_config: str,
        eventff_ckpt: str,
        img_size: tuple[int, int],
        Kl: list[float] | np.ndarray,
        Kr: list[float] | np.ndarray,
        distl: list[float] | np.ndarray,
        distr: list[float] | np.ndarray,
        r2l: np.ndarray,
        matcher_config: dict = None,
    ):
        """
        Initialize the stereo event matcher.
        """
        if matcher_config:
            self.StereoSGBMConfig = matcher_config
        else:
            self.StereoSGBMConfig = {
                "minDisparity": -64,
                "numDisparities": 128,
                "blockSize": 9,
                "P1": 8 * 3 * 9**2,
                "P2": 32 * 3 * 9**2,
                "disp12MaxDiff": 0,
                "uniquenessRatio": 5,
                "speckleWindowSize": 200,
                "speckleRange": 2,
                "mode": cv2.StereoSGBM_MODE_HH4,
            }

        self.r2l = r2l
        self.img_size = img_size
        self.distl = np.array(distl, dtype=np.float32)
        self.distr = np.array(distr, dtype=np.float32)
        self.Kl = np.array([[Kl[0], 0, Kl[2]], [0, Kl[1], Kl[3]], [0, 0, 1]])
        self.Kr = np.array([[Kr[0], 0, Kr[2]], [0, Kr[1], Kr[3]], [0, 0, 1]])

        # Load model
        self.eff_og = init_event_model(eventff_config, return_feat=True).cuda()
        self.eff = torch.compile(self.eff_og)
        epoch, loss, acc = load_weights_ckpt(self.eff, eventff_ckpt, strict=False)
        print(
            f"Feature Extractor :: {self.eff_og.__class__.__name__} :: Loaded model from epoch {epoch} with loss {loss} and accuracy {acc}"
        )

        self._setup_rectification()
        self._setup_stereo_matchers()

    def _setup_rectification(self):
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.Kl,
            self.distl,
            self.Kr,
            self.distr,
            self.img_size,
            self.r2l[:3, :3],
            self.r2l[:3, 3],
            alpha=0,
        )
        self.xmap1, self.ymap1 = cv2.initUndistortRectifyMap(
            self.Kl, self.distl, R1, P1, self.img_size, cv2.CV_32FC1
        )
        self.xmap2, self.ymap2 = cv2.initUndistortRectifyMap(
            self.Kr, self.distr, R2, P2, self.img_size, cv2.CV_32FC1
        )

    def _setup_stereo_matchers(self):
        """Setup stereo matching algorithms."""
        self.matcher = cv2.StereoSGBM_create(**self.StereoSGBMConfig)

    @torch.no_grad()
    def extract_features(self, ctx_left: torch.Tensor, ctx_right: torch.Tensor):
        cnt_left = torch.tensor([ctx_left.shape[0]], dtype=torch.int32).cuda()
        cnt_right = torch.tensor([ctx_right.shape[0]], dtype=torch.int32).cuda()
        feat_left = self.eff(ctx_left.cuda(), cnt_left)[1].detach().clone()
        feat_right = self.eff(ctx_right.cuda(), cnt_right)[1].detach().clone()
        return feat_left, feat_right

    def compute_disparity(self, ctx_left, ctx_right, use_pca=True, return_features=False):
        """
        Compute disparity map from event features at given timestamp.

        Args:
            ctx_left: Left context events (tensor)
            ctx_right: Right context events (tensor)
            use_pca: Whether to use PCA for feature visualization

        Returns:
            disparity: Disparity map
        """
        feat_left, feat_right = self.extract_features(ctx_left, ctx_right)

        feat_left = feat_left[0].cpu().numpy().transpose(1, 0, 2)  # 720, 1280, 32
        feat_right = feat_right[0].cpu().numpy().transpose(1, 0, 2)  # 720, 1280, 32

        if use_pca:
            pca = PCA(n_components=3)
            pca.fit(np.concatenate([feat_left.reshape(-1, 32), feat_right.reshape(-1, 32)], axis=0))
            pca_left = pca.transform(feat_left.reshape(-1, 32)).reshape(720, 1280, 3)
            pca_right = pca.transform(feat_right.reshape(-1, 32)).reshape(720, 1280, 3)

            min_val = np.min(np.concatenate([pca_left, pca_right], axis=0))
            max_val = np.max(np.concatenate([pca_left, pca_right], axis=0))
            pca_left = (pca_left - min_val) / (max_val - min_val)
            pca_right = (pca_right - min_val) / (max_val - min_val)
            pca_left = (pca_left * 255).astype(np.uint8)
            pca_right = (pca_right * 255).astype(np.uint8)

            pca_left_rect = cv2.remap(pca_left, self.xmap1, self.ymap1, cv2.INTER_LINEAR)
            pca_right_rect = cv2.remap(pca_right, self.xmap2, self.ymap2, cv2.INTER_LINEAR)

            ldisparity = self.matcher.compute(pca_left_rect, pca_right_rect).astype(np.float32)

            if return_features:
                return ldisparity, pca_left_rect, pca_right_rect
            return ldisparity
        else:
            # Use raw features for matching (you might need to adapt this)
            feat_left_rect = cv2.remap(feat_left, self.xmap1, self.ymap1, cv2.INTER_LINEAR)
            feat_right_rect = cv2.remap(feat_right, self.xmap2, self.ymap2, cv2.INTER_LINEAR)

            ldisparity = self.matcher.compute(feat_left_rect, feat_right_rect).astype(np.float32)

            if return_features:
                return ldisparity, pca_left_rect, pca_right_rect
            return ldisparity

    def process_disparity(self, disparity, colormap=cv2.COLORMAP_PLASMA, remove_outliers=True):
        """Process and colorize disparity map."""
        valid_pixels = disparity >= self.StereoSGBMConfig["minDisparity"]
        disparity[valid_pixels] = np.abs(disparity[valid_pixels])
        if remove_outliers:
            mu, sigma = np.mean(disparity[valid_pixels]), np.std(disparity[valid_pixels])
            valid_pixels[disparity > mu + 3 * sigma] = False
            valid_pixels[disparity < mu - 3 * sigma] = False

        valid_disparity = disparity[valid_pixels]
        min_disp = np.min(valid_disparity)
        max_disp = np.max(valid_disparity)
        disparity_colored = ((disparity - min_disp) * 255 / (max_disp - min_disp)).astype(np.uint8)

        disparity_colored = cv2.applyColorMap(disparity_colored, colormap)
        disparity_colored[~valid_pixels] = 0
        disparity_colored = cv2.cvtColor(disparity_colored, cv2.COLOR_BGR2RGB)

        return disparity_colored
