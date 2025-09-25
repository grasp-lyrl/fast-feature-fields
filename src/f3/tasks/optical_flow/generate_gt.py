""" Computes optical flow from two poses and depth images """
import cv2
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from scipy.linalg import logm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp, Rotation as R


class Flow:
    """
    - parameters
        - calibration :: a Calibration object from calibration.py
    """
    def __init__(self, calibration):
        self.cal = calibration

        self.P = np.array([[self.cal.intrinsics[0], 0., self.cal.intrinsics[2]],
                           [0., self.cal.intrinsics[1], self.cal.intrinsics[3]],
                           [0., 0., 1.]])
        resolution = self.cal.resolution

        x_inds, y_inds = np.meshgrid(np.arange(resolution[0]),
                                     np.arange(resolution[1]))
        x_inds = x_inds.astype(np.float32)
        y_inds = y_inds.astype(np.float32)

        x_inds -= self.P[0,2]
        x_inds *= (1./self.P[0,0])

        y_inds -= self.P[1,2]
        y_inds *= (1./self.P[1,1])

        self.flat_x_map = x_inds.reshape((-1))
        self.flat_y_map = y_inds.reshape((-1))

        N = self.flat_x_map.shape[0]

        self.omega_mat = np.zeros((N,2,3))

        self.omega_mat[:,0,0] = self.flat_x_map * self.flat_y_map
        self.omega_mat[:,1,0] = 1 + np.square(self.flat_y_map)

        self.omega_mat[:,0,1] = -(1 + np.square(self.flat_x_map))
        self.omega_mat[:,1,1] = -(self.flat_x_map * self.flat_y_map)

        self.omega_mat[:,0,2] = self.flat_y_map
        self.omega_mat[:,1,2] = -self.flat_x_map

    def compute_flow_single_frame(self, V, Omega, depth_image, dt):
        """
        params:
            V : [3,1]
            Omega : [3,1]
            depth_image : [m,n]
        """
        flat_depth = depth_image.ravel()
        # flat_depth[np.logical_or(np.isclose(flat_depth,0.0), flat_depth<0.)]
        mask = np.isfinite(flat_depth)

        fdm = 1./flat_depth[mask]
        fxm = self.flat_x_map[mask]
        fym = self.flat_y_map[mask]
        omm = self.omega_mat[mask,:,:]

        x_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_x_flow_out = x_flow_out.reshape((-1))
        flat_x_flow_out[mask] = fdm * (fxm*V[2]-V[0])
        flat_x_flow_out[mask] +=  np.squeeze(np.dot(omm[:,0,:], Omega))

        y_flow_out = np.zeros((depth_image.shape[0], depth_image.shape[1]))
        flat_y_flow_out = y_flow_out.reshape((-1))
        flat_y_flow_out[mask] = fdm * (fym*V[2]-V[1])
        flat_y_flow_out[mask] +=  np.squeeze(np.dot(omm[:,1,:], Omega))

        flat_x_flow_out *= dt * self.P[0,0]
        flat_y_flow_out *= dt * self.P[1,1]

        return x_flow_out, y_flow_out

    def interpolate_pose(self, H0, H1, alpha=0.5):
        # Extract rotation and translation components
        R0, t0 = H0[:3, :3], H0[:3, 3]
        R1, t1 = H1[:3, :3], H1[:3, 3]

        # Convert rotations to quaternions
        R_mat = R.from_matrix(np.stack([R0, R1], axis=0))

        slerp = Slerp([0, 1], R_mat)

        # Convert back to rotation matrix
        R_half = slerp([alpha]).as_matrix()

        # Interpolate translation linearly
        t_half = (1 - alpha) * t0 + alpha * t1

        # Construct the interpolated pose
        H_half = np.eye(4)
        H_half[:3, :3] = R_half
        H_half[:3, 3] = t_half

        return H_half
    
    def interpolate_depth(self, H0, H1, H_half, D0, D1):
        h, w = D0.shape

        mask0 = np.isfinite(D0).ravel()
        mask1 = np.isfinite(D1).ravel()

        X0 = np.stack([self.flat_x_map * D0.ravel(), self.flat_y_map * D0.ravel(), D0.ravel(), np.ones_like(self.flat_x_map)], axis=0) # [4, 720*1280]
        X1 = np.stack([self.flat_x_map * D1.ravel(), self.flat_y_map * D1.ravel(), D1.ravel(), np.ones_like(self.flat_x_map)], axis=0) # [4, 720*1280]
        X0 = X0[:, mask0]
        X1 = X1[:, mask1]
        
        X_0_half = H0 @ np.linalg.inv(H_half) @ X0
        X_1_half = H1 @ np.linalg.inv(H_half) @ X1

        D_half = np.zeros((h, w))
        X_0_half_proj = self.P @ X_0_half[:3]
        X_0_half_proj /= X_0_half_proj[2]
        X_0_half_proj = np.round(X_0_half_proj).astype(int)
        
        mask_0_half_proj = np.logical_and(np.logical_and(X_0_half_proj[0] >= 0, X_0_half_proj[0] < w),
                                          np.logical_and(X_0_half_proj[1] >= 0, X_0_half_proj[1] < h))
        
        D_half[X_0_half_proj[1, mask_0_half_proj], X_0_half_proj[0, mask_0_half_proj]] = X_0_half[2, mask_0_half_proj]
        D_half[D_half <= 0] = 0.0
        
        X_1_half_proj = self.P @ X_1_half[:3]
        X_1_half_proj /= X_1_half_proj[2]
        X_1_half_proj = np.round(X_1_half_proj).astype(int)
        
        mask_1_half_proj = np.logical_and(np.logical_and(X_1_half_proj[0] >= 0, X_1_half_proj[0] < w),
                                          np.logical_and(X_1_half_proj[1] >= 0, X_1_half_proj[1] < h))
        
        D_half[X_1_half_proj[1, mask_1_half_proj], X_1_half_proj[0, mask_1_half_proj]] = X_1_half[2, mask_1_half_proj]        
        
        return D_half

    def compute_velocity_from_poses(self, P0, P1, t0, t1):
        """
        Args:
            P0 :: [4,4] R|T 0|1
            P1 :: [4,4] R|T 0|1
        """
        H01 = np.dot(P0, np.linalg.inv(P1))
        dt = t1 - t0

        V = H01[:3, 3] / dt
        w_hat = logm(H01[:3, :3]) / dt
        Omega = np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

        return V, Omega, dt

    def flow_viz_np(self, flow, norm=False):
        h, w = flow.shape[:2]
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[:, :, 0] = ang * 180 / np.pi / 2
        hsv[:, :, 1] = 255
        if norm:
            magmean, magstd = np.mean(mag), np.std(mag)
            mag = np.clip(mag, magmean - 2 * magstd, magmean + 2 * magstd)
        hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return flow_rgb

    def colorize_image(self, flow_x, flow_y, norm=False):
        flow = np.stack((flow_x, flow_y), axis=2)
        return self.flow_viz_np(flow, norm)


def experiment_flow(cal, gt, base_name, save_movie=True,
                    save_dist=False, start_ind=None, stop_ind=None):
    """
        Args:
            cal :: Calibration object with intrinsics and extrinsics
            gt :: GroundTruth object with depth and odometry data
    """
    flow = Flow(cal)
    P0 = None

    nframes = len(gt.left_cam_readers['depth/prophesee/left'])
    if stop_ind is not None:
        stop_ind = min(nframes, stop_ind)
    else:
        stop_ind = nframes

    if start_ind is not None:
        start_ind = max(0, start_ind)
    else:
        start_ind = 0

    nframes = stop_ind - start_ind

    depth_image = gt.left_cam_readers['depth/prophesee/left'][0]
    flow_shape = (nframes, depth_image.shape[0], depth_image.shape[1])
    flow_chunk = (1, depth_image.shape[0], depth_image.shape[1])

    timestamps = np.zeros((nframes,), dtype=np.uint64)
    timestamps_map_prophesee_left = np.zeros((nframes,), dtype=np.uint64)

    Vs = np.zeros((nframes,3), dtype=float)
    Omegas = np.zeros((nframes,3), dtype=float)
    dTs = np.zeros((nframes,), dtype=float)

    # # show interpolated depth
    # for frame_num in range(100, nframes-1):
    #     P0 = gt.left_cam_readers['Cn_T_C0'][frame_num+start_ind]
    #     P1 = gt.left_cam_readers['Cn_T_C0'][frame_num+start_ind+1]
    #     P_half = flow.interpolate_pose(P0, P1, alpha=0.5)
    #     D0 = gt.left_cam_readers['depth/prophesee/left'][frame_num+start_ind]
    #     D1 = gt.left_cam_readers['depth/prophesee/left'][frame_num+start_ind+1]
    #     D_half = flow.interpolate_depth(P0, P1, P_half, D0, D1)

    #     cv2.imshow('depth', cv2.applyColorMap((D_half/np.max(D_half)*255).astype(np.uint8), cv2.COLORMAP_JET))
    #     cv2.waitKey(50)

    print("Extracting velocity")
    for frame_num in tqdm(range(nframes)):
        P1 = gt.left_cam_readers['Cn_T_C0'][frame_num+start_ind]
        t1 = gt.left_cam_readers['ts'][frame_num+start_ind] / 1e6 # in s

        if P0 is not None:
            V, Omega, dt = flow.compute_velocity_from_poses(P0, P1, t0, t1)
            Vs[frame_num, :] = V
            Omegas[frame_num, :] = Omega
            dTs[frame_num] = dt

        timestamps[frame_num] = gt.left_cam_readers['ts'][frame_num+start_ind] # in us
        timestamps_map_prophesee_left[frame_num] = gt.left_cam_readers['ts_map_prophesee_left'][frame_num+start_ind]

        P0 = P1.copy()
        t0 = t1.copy()

    filter_size = 10

    smoothed_Vs = Vs
    smoothed_Omegas = Omegas

    h5_name = base_name+"_gt_flow.h5"
    print("Computing flow and Saving to h5... ", h5_name)
    h5f_out = h5py.File(h5_name, 'w')
    h5f_out.create_dataset("ts", data=timestamps)
    h5f_out.create_dataset("ts_map_prophesee_left", data=timestamps_map_prophesee_left)

    h5f_out.create_dataset("flow/prophesee/left/x", shape=flow_shape, dtype=np.float32,
                           chunks=flow_chunk, compression='lzf')
    h5f_out.create_dataset("flow/prophesee/left/y", shape=flow_shape, dtype=np.float32,
                           chunks=flow_chunk, compression='lzf')

    minx, miny = 1e9, 1e9
    maxx, maxy = -1e9, -1e9

    for frame_num in tqdm(range(nframes)):
        depth_image = gt.left_cam_readers['depth/prophesee/left'][frame_num+start_ind]

        if frame_num-filter_size < 0:
            V = np.mean(Vs[0:frame_num+filter_size+1,:],axis=0)
            Omega = np.mean(Omegas[0:frame_num+filter_size+1,:], axis=0)
        elif frame_num+filter_size >= nframes:
            V = np.mean(Vs[frame_num-filter_size:nframes,:],axis=0)
            Omega = np.mean(Omegas[frame_num-filter_size:nframes,:], axis=0)
        else:
            V = np.mean(Vs[frame_num-filter_size:frame_num+filter_size+1,:],axis=0)
            Omega = np.mean(Omegas[frame_num-filter_size:frame_num+filter_size+1,:], axis=0)
        dt = dTs[frame_num]

        smoothed_Vs[frame_num, :] = V
        smoothed_Omegas[frame_num, :] = Omega

        flow_x_dist, flow_y_dist = flow.compute_flow_single_frame(V, Omega, depth_image, dt)
        h5f_out['flow/prophesee/left/x'][frame_num] = flow_x_dist
        h5f_out['flow/prophesee/left/y'][frame_num] = flow_y_dist

        minx, miny = min(minx, np.min(flow_x_dist)), min(miny, np.min(flow_y_dist))
        maxx, maxy = max(maxx, np.max(flow_x_dist)), max(maxy, np.max(flow_y_dist))        

    if save_movie:
        print("Saving movie")
        out = cv2.VideoWriter(base_name+"_gt_flow.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (1280, 720))
        for frame_num in range(nframes):
            flow_img = flow.colorize_image(h5f_out['flow/prophesee/left/x'][frame_num],
                                           h5f_out['flow/prophesee/left/y'][frame_num], norm=True)
            out.write(flow_img)
        out.release()

    if save_dist: # save the distribution of pixel displacement
        print("Saving distribution")
        def plot_hist(dir='x', bins=50, chunk_size=100):
            min_val, max_val = minx if dir == 'x' else miny, maxx if dir == 'x' else maxy
            hist_bins = np.linspace(min_val, max_val, bins+1)
            hist_counts = np.zeros(bins)

            for i in range(0, h5f_out[f'flow/prophesee/left/{dir}'].shape[0], chunk_size):
                chunk = h5f_out[f'flow/prophesee/left/{dir}'][i : i + chunk_size]
                chunk = chunk[np.abs(chunk) > 1.0]
                hist_counts += np.histogram(chunk, bins=hist_bins)[0]

            plt.figure(figsize=(8, 6))
            plt.bar(hist_bins[:-1], hist_counts, width=np.diff(hist_bins), edgecolor="black", alpha=0.7)
            plt.xlabel("Pixel displacement")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of {dir} flow")
            plt.grid(True)
            plt.savefig(base_name+f"_{dir}_hist.png", dpi=600)
            plt.close()
        plot_hist('x')
        plot_hist('y')
    h5f_out.close()


parser = argparse.ArgumentParser()
parser.add_argument('--events_h5',
                    type=str,
                    help="Path to M3ED events h5 file.",
                    required=True)
parser.add_argument('--depth_h5',
                    type=str,
                    help="Path to M3ED depth h5 file.",
                    required=True)
parser.add_argument('--base_name',
                    type=str,
                    help="Base name for the output files.")
parser.add_argument('--save_movie',
                    action='store_true',
                    help="If set, will save a movie of the estimated flow for visualization.")
parser.add_argument('--save_dist',
                    action='store_true',
                    help="If set, will save the results to a numpy file.")
parser.add_argument('--start_ind',
                    type=int,
                    help="Index of the first ground truth pose/depth frame to process.",
                    default=None)
parser.add_argument('--stop_ind',
                    type=int,
                    help="Index of the last ground truth pose/depth frame to process.",
                    default=None)
args = parser.parse_args()


class GroundTruth:
    def __init__(self, depth_h5_path):
        self.depth_h5_path = depth_h5_path
        self.depth_h5 = h5py.File(self.depth_h5_path, 'r')

        topics = ['Cn_T_C0', 'depth/prophesee/left', 'ts', 'ts_map_prophesee_left']
        self.left_cam_readers = {}
        for t in topics:
            self.left_cam_readers[t] = self.depth_h5[t]


class Calibration:
    def __init__(self, events_h5_path):
        self.events_h5_path = events_h5_path
        self.events_h5 = h5py.File(self.events_h5_path, 'r')
        self.resolution = self.events_h5['prophesee/left/calib/resolution'][:]
        self.intrinsics = self.events_h5['prophesee/left/calib/intrinsics'][:]


if __name__=="__main__":
    cal = Calibration(args.events_h5)
    gt = GroundTruth(args.depth_h5)

    if args.base_name is None:
        args.base_name = str.replace(args.events_h5, "_data.h5", "")

    experiment_flow(cal, gt, args.base_name, args.save_movie, args.save_dist, args.start_ind, args.stop_ind)
