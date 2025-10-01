#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate f3

base_path=$1

ln -nsf $base_path/falcon_indoor_flight_1 data/
ln -nsf $base_path/falcon_indoor_flight_2 data/
ln -nsf $base_path/falcon_indoor_flight_3 data/

ln -nsf $base_path/falcon_outdoor_day_fast_flight_1 data/
ln -nsf $base_path/falcon_outdoor_day_fast_flight_2 data/
ln -nsf $base_path/falcon_outdoor_day_fast_flight_3 data/
ln -nsf $base_path/falcon_outdoor_day_penno_cars data/
ln -nsf $base_path/falcon_outdoor_day_penno_parking_1 data/
ln -nsf $base_path/falcon_outdoor_day_penno_parking_2 data/
ln -nsf $base_path/falcon_outdoor_day_penno_parking_3 data/
ln -nsf $base_path/falcon_outdoor_day_penno_plaza data/
ln -nsf $base_path/falcon_outdoor_day_penno_trees data/

ln -nsf $base_path/falcon_outdoor_night_high_beams data/
ln -nsf $base_path/falcon_outdoor_night_penno_parking_1 data/
ln -nsf $base_path/falcon_outdoor_night_penno_parking_2 data/


# Generate timestamps for the M3ED dataset
python3 scripts/generate_ts.py --data_h5 data/falcon_indoor_flight_1/falcon_indoor_flight_1_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_indoor_flight_2/falcon_indoor_flight_2_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_indoor_flight_3/falcon_indoor_flight_3_data.h5 --dataset "m3ed"

python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_day_fast_flight_1/falcon_outdoor_day_fast_flight_1_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_day_fast_flight_2/falcon_outdoor_day_fast_flight_2_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_day_fast_flight_3/falcon_outdoor_day_fast_flight_3_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_day_penno_cars/falcon_outdoor_day_penno_cars_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_day_penno_parking_1/falcon_outdoor_day_penno_parking_1_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_day_penno_parking_2/falcon_outdoor_day_penno_parking_2_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_day_penno_parking_3/falcon_outdoor_day_penno_parking_3_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_day_penno_plaza/falcon_outdoor_day_penno_plaza_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_day_penno_trees/falcon_outdoor_day_penno_trees_data.h5 --dataset "m3ed"

python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_night_high_beams/falcon_outdoor_night_high_beams_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_night_penno_parking_1/falcon_outdoor_night_penno_parking_1_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/falcon_outdoor_night_penno_parking_2/falcon_outdoor_night_penno_parking_2_data.h5 --dataset "m3ed"
