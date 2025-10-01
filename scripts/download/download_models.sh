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
  echo "Coming soon!"
fi

if [ "$TYPE" == "depth" ]; then
  echo "Coming soon!"
fi

if [ "$TYPE" == "flow" ]; then
  echo "Coming soon!"
fi

echo "Download completed."
