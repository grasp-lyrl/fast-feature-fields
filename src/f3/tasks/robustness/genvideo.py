import cv2
import h5py
import torch
import numpy as np
from tqdm import tqdm
from numpy.random import default_rng

rng = default_rng(42)

from f3 import init_event_model, load_weights_ckpt
from f3.utils import setup_torch, plot_patched_features, BaseExtractor, ev_to_frames, smooth_time_weighted_rgb_encoding

setup_torch(cudnn_benchmark=False)

time = int(200 * 1e6) # in us
img_size = (1280, 720)
width, height = img_size

# config = "/home/richeek/GitHub/f3/outputs/patchff_fullcardaym3ed_small_20ms/models/config.yml"
# ckpt = "/home/richeek/GitHub/f3/outputs/patchff_fullcardaym3ed_small_20ms/models/last.pth"

config = "/home/richeek/GitHub/f3/outputs/patchff_fullcardaynightm3ed_small_20ms/models/config.yml"
ckpt = "/home/richeek/GitHub/f3/outputs/patchff_fullcardaynightm3ed_small_20ms/models/last.pth"

# h5path = "/home/richeek/GitHub/f3/data/car_urban_day_city_hall_data/car_urban_day_city_hall_data.h5"
# tspath = "/home/richeek/GitHub/f3/data/car_urban_day_city_hall_data/50khz_car_urban_day_city_hall_data.npy"
# h5path = "/home/richeek/GitHub/f3/data/car_urban_night_city_hall_data/car_urban_night_city_hall_data.h5"
# tspath = "/home/richeek/GitHub/f3/data/car_urban_night_city_hall_data/50khz_car_urban_night_city_hall_data.npy"

# h5path = "/home/richeek/GitHub/f3/data/car_urban_day_rittenhouse_data/car_urban_day_rittenhouse_data.h5"
# tspath = "/home/richeek/GitHub/f3/data/car_urban_day_rittenhouse_data/50khz_car_urban_day_rittenhouse_data.npy"
# h5path = "/home/richeek/GitHub/f3/data/car_urban_night_rittenhouse_data/car_urban_night_rittenhouse_data.h5"
# tspath = "/home/richeek/GitHub/f3/data/car_urban_night_rittenhouse_data/50khz_car_urban_night_rittenhouse_data.npy"

# h5path = "/home/richeek/GitHub/f3/data/car_urban_day_penno_small_loop_data/car_urban_day_penno_small_loop_data.h5"
# tspath = "/home/richeek/GitHub/f3/data/car_urban_day_penno_small_loop_data/50khz_car_urban_day_penno_small_loop_data.npy"

# h5path = "/home/richeek/GitHub/f3/data/car_urban_day_ucity_small_loop_data/car_urban_day_ucity_small_loop_data.h5"
# tspath = "/home/richeek/GitHub/f3/data/car_urban_day_ucity_small_loop_data/50khz_car_urban_day_ucity_small_loop_data.npy"
h5path = "/home/richeek/GitHub/f3/data/car_urban_night_ucity_small_loop_data/car_urban_night_ucity_small_loop_data.h5"
tspath = "/home/richeek/GitHub/f3/data/car_urban_night_ucity_small_loop_data/50khz_car_urban_night_ucity_small_loop_data.npy"

# h5path = "/home/richeek/GitHub/f3/data/car_urban_day_schuylkill_tunnel_data/car_urban_day_schuylkill_tunnel_data.h5"
# tspath = "/home/richeek/GitHub/f3/data/car_urban_day_schuylkill_tunnel_data/50khz_car_urban_day_schuylkill_tunnel_data.npy"

dataname = h5path.split("/")[-1].split(".")[0]
print(f"Processing {dataname}...")

h5file = h5py.File(h5path, 'r')
tsfile = np.load(tspath, allow_pickle=True)[()]

extractor = BaseExtractor(h5path, tspath, w=1280, h=720, time_ctx=20000,
                          time_pred=20000, bucket=1000, max_numevents_ctx=2000000,
                          randomize_ctx=False, camera="left")

eff = init_event_model(config, return_feat=True).cuda()
eff = torch.compile(eff)

epoch, loss, acc = load_weights_ckpt(eff, ckpt, strict=False)

print(f"Loaded model from epoch {epoch} with loss {loss} and accuracy {acc}")


# create video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(f"{dataname}_video.mp4", fourcc, 25.0, (1280*2, 720*2))

for rgb_idx in tqdm(range(0, len(h5file["ovc/rgb/data"]))):
    time = h5file["ovc/ts"][rgb_idx]  # in us
    if time <= 20000: continue  # skip first 20ms

    rgb = h5file["ovc/rgb/data"][rgb_idx]
    ctx, totcnt = extractor.get_ctx_fixedtime(time)

    if totcnt < 50000: continue  # skip frames with less than 100k events
    eframe = ev_to_frames(ctx, torch.tensor([totcnt], dtype=torch.int32), *img_size)[0].T.cpu().numpy()[..., None].repeat(3, axis=-1)

    ctx_tensor = ctx.cuda()
    cnt_tensor = torch.tensor([totcnt], dtype=torch.int32).cuda()

    with torch.no_grad():
        logits, feats = eff(ctx_tensor, cnt_tensor)
        predframe = smooth_time_weighted_rgb_encoding(torch.sigmoid(logits).cpu().numpy() > 0.6225)[0].transpose(1, 0, 2)

    pca, _ = plot_patched_features(feats[0].permute(1, 0, 2), plot=False)

    combined = np.zeros((720 * 2, 1280 * 2, 3), dtype=np.uint8)
    combined[:720, :1280] = rgb[:720]
    combined[:720, 1280:] = eframe
    combined[720:, :1280] = pca
    combined[720:, 1280:] = predframe

    combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    out.write(combined)
out.release()