#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate f3

base_path=$1

ln -nsf $base_path/spot_indoor_building_loop data/
ln -nsf $base_path/spot_indoor_obstacles data/
ln -nsf $base_path/spot_indoor_stairs data/
ln -nsf $base_path/spot_indoor_stairwell data/

ln -nsf $base_path/spot_outdoor_day_art_plaza_loop data/
ln -nsf $base_path/spot_outdoor_day_penno_building_loop data/
ln -nsf $base_path/spot_outdoor_day_penno_short_loop data/
ln -nsf $base_path/spot_outdoor_day_rocky_steps data/
ln -nsf $base_path/spot_outdoor_day_skatepark_1 data/
ln -nsf $base_path/spot_outdoor_day_skatepark_2 data/
ln -nsf $base_path/spot_outdoor_day_skatepark_3 data/
ln -nsf $base_path/spot_outdoor_day_srt_green_loop data/
ln -nsf $base_path/spot_outdoor_day_srt_under_bridge_1 data/
ln -nsf $base_path/spot_outdoor_day_srt_under_bridge_2 data/

ln -nsf $base_path/spot_outdoor_night_penno_building_loop data/
ln -nsf $base_path/spot_outdoor_night_penno_plaza_lights data/
ln -nsf $base_path/spot_outdoor_night_penno_short_loop data/

# Generate timestamps for the M3ED dataset
python3 scripts/generate_ts.py --data_h5 data/spot_indoor_building_loop/spot_indoor_building_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_indoor_obstacles/spot_indoor_obstacles_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_indoor_stairs/spot_indoor_stairs_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_indoor_stairwell/spot_indoor_stairwell_data.h5 --dataset "m3ed"

python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_day_art_plaza_loop/spot_outdoor_day_art_plaza_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_day_penno_building_loop/spot_outdoor_day_penno_building_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_day_penno_short_loop/spot_outdoor_day_penno_short_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_day_rocky_steps/spot_outdoor_day_rocky_steps_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_day_skatepark_1/spot_outdoor_day_skatepark_1_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_day_skatepark_2/spot_outdoor_day_skatepark_2_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_day_skatepark_3/spot_outdoor_day_skatepark_3_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_day_srt_green_loop/spot_outdoor_day_srt_green_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_day_srt_under_bridge_1/spot_outdoor_day_srt_under_bridge_1_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_day_srt_under_bridge_2/spot_outdoor_day_srt_under_bridge_2_data.h5 --dataset "m3ed"

python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_night_penno_building_loop/spot_outdoor_night_penno_building_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_night_penno_plaza_lights/spot_outdoor_night_penno_plaza_lights_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/spot_outdoor_night_penno_short_loop/spot_outdoor_night_penno_short_loop_data.h5 --dataset "m3ed"
