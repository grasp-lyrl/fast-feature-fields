# Dataset Setup

This repository uses the following datasets: M3ED, DSEC and MVSEC. If you download these datasets in some location, you can use the scripts provided in `scripts/setup_<dataset_name>.sh` to create symbolic links to these datasets in the `data/` folder. These scripts also create the required `timestamps.npy` files. 

---

### M3ED

```bash
bash scripts/setup_m3ed_car.sh /path/to/M3ED/
bash scripts/setup_m3ed_spot.sh /path/to/M3ED/
bash scripts/setup_m3ed_falcon.sh /path/to/M3ED/
```

### DSEC

```bash
bash scripts/setup_dsec.sh /path/to/DSEC/
```

---

#### TODO

- [ ] Remove the need for `.npy` files. These have remained to support finer time discretization of events than 1 ms.