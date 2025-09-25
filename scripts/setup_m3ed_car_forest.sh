#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate f3

base_path=$1

ln -nsf $base_path/car_forest_into_ponds_long data/
ln -nsf $base_path/car_forest_into_ponds_short data/
ln -nsf $base_path/car_forest_sand_1 data/
ln -nsf $base_path/car_forest_sand_2 data/
ln -nsf $base_path/car_forest_tree_tunnel data/

# Generate timestamps for the M3ED dataset
python3 scripts/generate_ts.py --data_h5 data/car_forest_into_ponds_long/car_forest_into_ponds_long_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_forest_into_ponds_short/car_forest_into_ponds_short_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_forest_sand_1/car_forest_sand_1_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_forest_sand_2/car_forest_sand_2_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_forest_tree_tunnel/car_forest_tree_tunnel_data.h5 --dataset "m3ed"
