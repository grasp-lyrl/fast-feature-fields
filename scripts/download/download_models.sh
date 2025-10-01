#!/bin/bash

DEST=$1
TYPE=$2
BASE_URL="https://richeek-penn.s3.amazonaws.com/prolev"

if [ -z "$DEST" ] || [ -z "$TYPE" ]; then
  echo "Usage: $0 <destination_directory> <model_type>"
  echo "Model types: f3, seg, depth, flow"
  exit 1
fi

if [ ! -d "$DEST" ]; then
  echo "Creating destination directory: $DEST"
  mkdir -p "$DEST"
fi

if [ "$TYPE" == "f3" ]; then
  echo "Downloading f3 models to $DEST"
  wget -P "$DEST" "$BASE_URL"/ff/patchff_fullcardaym3ed_small_20ms.pth
  wget -P "$DEST" "$BASE_URL"/ff/patchff_fullcardaym3ed_micro_20ms.pth
  wget -P "$DEST" "$BASE_URL"/ff/patchff_fullcardaym3ed_tiny_20ms.pth
  wget -P "$DEST" "$BASE_URL"/ff/patchff_fulldsec_small_20ms.pth
  wget -P "$DEST" "$BASE_URL"/ff/patchff_fulloutdoormvsec_small_50ms.pth
fi

if [ "$TYPE" == "seg" ]; then
  echo "Downloading semantic segmentation models to $DEST"
  MODEL="segformer_b3_fullm3ed_800x600x20"
  mkdir -p "$DEST"/"$MODEL"
  wget -P "$DEST"/"$MODEL" "$BASE_URL"/segmentation/"$MODEL"/models/best_miou.pth
  wget -P "$DEST"/"$MODEL" "$BASE_URL"/segmentation/"$MODEL"/models/segmentation_config.yml
fi

if [ "$TYPE" == "depth" ]; then
  echo "Downloading pseudo monocular depth models to $DEST"
  MODEL="dav2b_fullm3ed_pseudo_518x518x20"
  mkdir -p "$DEST"/"$MODEL"
  wget -P "$DEST"/"$MODEL" "$BASE_URL"/monoculardepth/"$MODEL"/best.pth
  wget -P "$DEST"/"$MODEL" "$BASE_URL"/monoculardepth/"$MODEL"/depth_config.yml
fi

if [ "$TYPE" == "flow" ]; then
  echo "Downloading optical flow models to $DEST"
  MODEL="optflow_trainm3ed_20msff_pyr5_28k"
  mkdir -p "$DEST"/"$MODEL"
  wget -P "$DEST"/"$MODEL" "$BASE_URL"/flow/"$MODEL"/last.pth
  wget -P "$DEST"/"$MODEL" "$BASE_URL"/flow/"$MODEL"/flow_config.yaml
fi

echo "Download completed."
