#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate f3

base_path=$1

ln -nsf $base_path/car_urban_day_city_hall data/
ln -nsf $base_path/car_urban_day_penno_big_loop data/
ln -nsf $base_path/car_urban_day_penno_small_loop data/
ln -nsf $base_path/car_urban_day_horse data/
ln -nsf $base_path/car_urban_day_rittenhouse data/
ln -nsf $base_path/car_urban_day_ucity_big_loop data/
ln -nsf $base_path/car_urban_day_ucity_small_loop data/
ln -nsf $base_path/car_urban_day_schuylkill_tunnel data/
ln -nsf $base_path/car_urban_night_city_hall data/
ln -nsf $base_path/car_urban_night_penno_big_loop data/
ln -nsf $base_path/car_urban_night_penno_small_loop data/
ln -nsf $base_path/car_urban_night_penno_small_loop_darker data/
ln -nsf $base_path/car_urban_night_rittenhouse data/
ln -nsf $base_path/car_urban_night_ucity_small_loop data/

# Generate timestamps for the M3ED dataset
python3 scripts/generate_ts.py --data_h5 data/car_urban_day_city_hall/car_urban_day_city_hall_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_day_penno_big_loop/car_urban_day_penno_big_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_day_penno_small_loop/car_urban_day_penno_small_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_day_horse/car_urban_day_horse_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_day_rittenhouse/car_urban_day_rittenhouse_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_day_ucity_big_loop/car_urban_day_ucity_big_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_day_ucity_small_loop/car_urban_day_ucity_small_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_day_schuylkill_tunnel/car_urban_day_schuylkill_tunnel_data.h5 --dataset "m3ed"

python3 scripts/generate_ts.py --data_h5 data/car_urban_night_city_hall/car_urban_night_city_hall_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_night_penno_big_loop/car_urban_night_penno_big_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_night_penno_small_loop/car_urban_night_penno_small_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_night_penno_small_loop_darker/car_urban_night_penno_small_loop_darker_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_night_rittenhouse/car_urban_night_rittenhouse_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_night_ucity_small_loop/car_urban_night_ucity_small_loop_data.h5 --dataset "m3ed"
python3 scripts/generate_ts.py --data_h5 data/car_urban_night_ucity_big_loop/car_urban_night_ucity_big_loop_data.h5 --dataset "m3ed"
