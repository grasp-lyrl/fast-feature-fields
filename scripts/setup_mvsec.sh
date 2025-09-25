#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate f3

base_path=$1

mkdir -p data/outdoor_day1_data
mkdir -p data/outdoor_day2_data
mkdir -p data/outdoor_night1_data

cd data/outdoor_day1_data
ln -sf $base_path/outdoor_day/outdoor_day1_processed.hdf5 outdoor_day1_data.hdf5
ln -sf $base_path/outdoor_day/outdoor_day1_gt.hdf5 .

cd ../outdoor_day2_data
ln -sf $base_path/outdoor_day/outdoor_day2_processed.hdf5 outdoor_day2_data.hdf5
ln -sf $base_path/outdoor_day/outdoor_day2_gt.hdf5 .

cd ../outdoor_night1_data
ln -sf $base_path/outdoor_night/outdoor_night1_processed.hdf5 outdoor_night1_data.hdf5
ln -sf $base_path/outdoor_night/outdoor_night1_gt.hdf5 .

cd ../..

# Generate timestamps for the M3ED dataset
python3 scripts/generate_ts.py --data_h5 data/outdoor_day1_data/outdoor_day1_data.hdf5 --dataset "mvsec"
python3 scripts/generate_ts.py --data_h5 data/outdoor_day2_data/outdoor_day2_data.hdf5 --dataset "mvsec"
python3 scripts/generate_ts.py --data_h5 data/outdoor_night1_data/outdoor_night1_data.hdf5 --dataset "mvsec"
