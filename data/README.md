# Dataset Setup

This repository uses the following datasets: M3ED, DSEC and MVSEC. If you download these datasets in some location, you can use the scripts provided in `scripts/setup/setup_<dataset_name>.sh` to create symbolic links to these datasets in the `data/` folder. These scripts also create the required `50kHz_timestamps.npy` files. 

---

### M3ED

The M3ED dataset is present [here](https://m3ed.io/). To download the necessary sequences for `car`, `spot`, and `falcon`, use the script [`download_m3ed.py`](/scripts/download/download_m3ed.py). Run the following scripts to download the sequences:

```bash
# Car urban driving sequences
python3 scripts/download/download_m3ed.py --vehicle car --environment urban --output_dir /path/to/M3ED/

# Spot indoor and outdoor sequences
python3 scripts/download/download_m3ed.py --vehicle spot --environment indoor --output_dir /path/to/M3ED/
python3 scripts/download/download_m3ed.py --vehicle spot --environment outdoor --output_dir /path/to/M3ED/

# Falcon indoor and outdoor sequences
python3 scripts/download/download_m3ed.py --vehicle falcon --environment indoor --output_dir /path/to/M3ED/
python3 scripts/download/download_m3ed.py --vehicle falcon --environment outdoor --output_dir /path/to/M3ED/
```

After downloading the necessary sequences, run the following commands to set up the symbolic links in `data/` and `50kHz_timestamps.npy` files:


```bash
bash scripts/setup_m3ed_car.sh /path/to/M3ED/
bash scripts/setup_m3ed_spot.sh /path/to/M3ED/
bash scripts/setup_m3ed_falcon.sh /path/to/M3ED/
```

**Note:** If you followed the instructions above to download the `car` sequences, you can train an F<sup>3</sup> model on the `car urban daytime driving` sequences using the command shown in the [Training an F<sup>3</sup>](#training-an-f3) section of the [README](/README.md). Setting up F<sup>3</sup> training for other M3ED sequences is straightforward -- choose the appropriate configuration file from `confs/ff/trainoptions/` and enjoy!

### DSEC

Download the DSEC dataset from [here](https://dsec.ifi.uzh.ch/dsec-datasets/download/). After downloading and extracting the dataset, run the following command to set up the symbolic links and `timestamps.npy` files:

```bash
bash scripts/setup_dsec.sh /path/to/DSEC/
```

### MVSEC

Download the MVSEC dataset from [here](https://drive.google.com/drive/folders/1rwyRk26wtWeRgrAx_fgPc-ubUzTFThkV). After downloading the `.hdf5` files, run:

```bash
python3 scripts/process_mvsec.py /path/to/MVSEC/<XYZ>.hdf5 /path/to/MVSEC/<XYZ>_processed.hdf5
```

for the sequences `outdoor_day1`, `outdoor_day2`, and `outdoor_night1`. Then run:

```bash
bash scripts/setup_mvsec.sh /path/to/MVSEC/
```



---

#### TODO

- [ ] Remove the need for `.npy` files. These have remained to support finer time discretization of events than 1 ms.