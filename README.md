# Fast Feature Field (F3): A Predictive Representation of Events


## Installation

```bash
conda create -n f3 python=3.11
conda activate f3
```

To install F3 locally:

```bash
git clone git@github.com:grasp-lyrl/fast-feature-fields.git
cd fast-feature-fields
pip install -e .
```

## Quickstart


## Training

```bash
accelerate launch --config_file confs/accelerate_confs/2GPU.yml main.py\
                  --conf confs/ff/trainoptions/patchff_fullcardaym3ed_small_20ms.yml\
                  --compile
```