# Fast Feature Field (F<sup>3</sup>): A Predictive Representation of Events

<div align="center">

![F<sup>3</sup> Logo](assets/figure1.webp)

[Richeek Das](https://www.seas.upenn.edu/~richeek/), [Kostas Daniilidis](https://www.cis.upenn.edu/~kostas/), [Pratik Chaudhari](https://pratikac.github.io/)

*GRASP Laboratory, University of Pennsylvania*

[[📜 Paper](https://arxiv.org/abs/2509.25146)] • [[🎬 Video](#)] • [[🌐 Website](https://www.seas.upenn.edu/~richeek/f3/)] • [[📖 BibTeX](#citation)]

</div>

## Overview

F<sup>3</sup> architecture is designed specifically for events. F<sup>3</sup> is a predictive representation of events. It is a statistic of past events sufficient to predict future events. We prove that such a representation retains information about the structure and motion in the scene. F<sup>3</sup> achieves low-latency computation by exploiting the sparsity of event data using a multi-resolution hash encoder and permutation-invariant architecture. Our implementation can compute F<sup>3</sup> at 120 Hz and 440 Hz at HD and VGA resolutions, respectively, and can predict different downstream tasks at 25-75 Hz at HD resolution. These HD inference rates are roughly 2-5 times faster than the current state-of-the-art event-based methods. Please refer to the [paper](https://arxiv.org/abs/2509.25146) for more details.

<div align="center">

![F3 Architecture](assets/arch.webp)

*An overview of the neural architecture for Fast Feature Field (F<sup>3</sup>) its downstream variants.*

</div>


## Quickstart

### Installation

```bash
conda create -n f3 python=3.11
conda activate f3
```

Install F<sup>3</sup> locally:

```bash
git clone git@github.com:grasp-lyrl/fast-feature-fields.git
cd fast-feature-fields
pip install -e .
```

### Inference using pretrained F<sup>3</sup>

To get you up and running quickly, we can download a small sequence from M3ED and run some inference tasks on it.

### Training an F<sup>3</sup>

Please refer to `data/README.md` for detailed instructions on setting up the datasets. This is only important if you want to train F<sup>3</sup> models on the M3ED, DSEC or MVSEC datasets.

```bash
accelerate launch --config_file confs/accelerate_confs/2GPU.yml main.py\
                  --conf confs/ff/trainoptions/patchff_fullcardaym3ed_small_20ms.yml\
                  --compile
```

### Citation

If you find this code useful in your research, please consider citing:

```bibtex
@misc{das2025fastfeaturefield,
  title={Fast Feature Field ($\text{F}^3$): A Predictive Representation of Events}, 
  author={Richeek Das and Kostas Daniilidis and Pratik Chaudhari},
  year={2025},
  eprint={2509.25146},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2509.25146},
}
