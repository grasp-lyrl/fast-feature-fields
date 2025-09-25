"""
Plots 4 results and GT and other stuff -> saves as video
"""
import re
import cv2
import argparse
import subprocess
import numpy as np
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Make video from images')
parser.add_argument('--base_path', type=str, default='outputs', help='Base path to the results')
parser.add_argument('--paths', nargs='+', type=str, default=[
    'predictions/mlplastlayer_*_*_pca.png',
    'predictions/shift_*_*.png',
    'predictions/hashed_*_*_pca.png',
    'events/shift_*_*.png',
    'predictions/loss_*_*.png',
    'predictions/predicted_full_*_*.png',
], help='Paths to images')
parser.add_argument('--fps', type=int, default=15, help='Frames per second')
parser.add_argument('--compress', action='store_true', help='Compress the video')
parser.add_argument('--start', type=int, default=0, help='Start index')
parser.add_argument('--end', type=int, default=-1, help='End index')
args = parser.parse_args()

def extract_numbers(filename):
    numbers = re.findall(r'\d+', filename)
    return tuple(map(int, numbers))

# Function to load an image, resize it, and place it in a specified quadrant of a frame
def process_image_quadrant(image_path, cx, cy, frame, width, height):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image {image_path} failed to load.")
    resized_img = cv2.resize(img, (width, height))
    lh, hh = cx * height, (cx + 1) * height
    lw, hw = int(cy * width), int((cy + 1) * width)
    frame[lh:hh, lw:hw] = resized_img
    return frame

# Video parameters
fps = args.fps  # Frames per second
results_path = args.base_path + '/'

nfolders = len(args.paths) # Number of folders to plot Supported 2, 3, 4, 5, 6
rows, cols = 2, (nfolders + 1) // 2

plotlist = []
for path in args.paths:
    content = sorted(glob(results_path + path), key=extract_numbers)
    plotlist.append(content)

if args.end == -1:
    args.end = len(plotlist[0])

for i in range(nfolders):
    plotlist[i] = plotlist[i][args.start:args.end]


height, width = 720, 1280  # Height and width of each quadrant
video_filename = f'{args.base_path}/results_{args.start}_{args.end}_{nfolders}panel.mp4'
out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * cols, height * rows))

for i in tqdm(range(args.end - args.start)):
    frame = np.zeros((height * rows, width * cols, 3), np.uint8)
    for j, path in enumerate(args.paths):
        cx, cy = j // cols, j % cols
        if nfolders % 2 == 1 and j == nfolders - 1: cy += 0.5
        frame = process_image_quadrant(plotlist[j][i], cx, cy, frame, width, height)
    out.write(frame)
out.release()


# run ffmpeg using popen
# if args.compress:
#     command = f"ffmpeg -i {video_filename} -vcodec libx265 -crf 35 outputs/{args.name}/test/{args.model}/{args.set}_results_compressed.mp4"
#     process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

# ffmpeg -ss 5 -i test_results.mp4 -t 10 -c:v libx265 -crf 35 out.mp4
