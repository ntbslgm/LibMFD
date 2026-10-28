# Transfer Learning for Fault Diagnosis



#### Repository: https://github.com/ntbslgm/LibMFD


## Overview of the Repository

This repository contains the loading code for **12 fault-diagnosis datasets** and the implementation
of **27 methods** (26 domain-adaptation / domain-generalization methods + 1 CNN baseline) for
cross-domain fault diagnosis of rotating machinery. We focus on three related settings —
**Single-source Unsupervised Domain Adaptation (SUDA)**, **Multi-source Unsupervised Domain
Adaptation (MUDA)**, and **Domain Generalization (DG)** — under a unified data preprocessing and
training pipeline. Every method is implemented against the same interface (`models/<Method>.py`) and
trained with the same optimizer, scheduler, and evaluation protocol, so that the reported accuracy
and F1 scores are directly comparable across methods.

## Benchmark Pipeline

Input raw vibration signals (`.mat`, `.csv`, or `.txt`) are preprocessed with a unified pipeline
before being passed to each model:

1. **Signal segmentation** — each raw signal is split into fixed-length samples with a sliding window
   (`--signal_size`, default 1024, no overlap).
2. **Normalization** — samples are normalized with one of `-1-1` (default), `0-1`, or `mean-std`
   (`--normlizetype`).
3. **Source/target split** — the labeled source domain is used entirely for training; the unlabeled
   target domain is split into a training subset and a validation subset (`test_size=0.2`).
4. **Model** — a feature extractor (multi-scale 1-D CNN by default, or 1-D ResNet via the
   `_ResNet` suffix) followed by a task classifier, plus method-specific domain-alignment modules.
5. **Training** — supervised cross-entropy loss on the source, combined with domain-adversarial /
   alignment losses (DA methods) or generalization regularizers (DG methods).
6. **Evaluation** — target-domain **accuracy** and **F1-score** are tracked over epochs and the
   best-epoch results are reported.

## Methods

We benchmark **27 methods** for cross-domain fault diagnosis, grouped into domain adaptation,
domain generalization, and a plain CNN baseline. Each method is implemented in `models/` and is
trained under the same data preprocessing and training pipelines for fair comparison. The code for
each method has been adapted from the referenced open-source implementations to the unified
`train.py` interface.

### Domain Adaptation

- **ACDANN** - Integrating expert knowledge with domain adaptation for unsupervised fault diagnosis. [TIM 2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9612159) | [View Code](/models/ACDANN.py)
- **ADACL** - Adversarial domain adaptation with classifier alignment for cross-domain intelligent fault diagnosis of multiple source domains. [Measurement Science and Technology 2020](https://iopscience.iop.org/article/10.1088/1361-6501/abcad4/pdf) | [View Code](/models/ADACL.py)
- **ADAST** - Attentive cross-domain framework with iterative self-training. [IEEE TETCI 2023] | [Reference Code](https://github.com/emadeldeen24/ADAST) | [View Code](/models/ADAST.py)
- **BSP** - Transferability vs. discriminability: Batch spectral penalization for adversarial domain adaptation. [ICML 2019](http://proceedings.mlr.press/v97/chen19i/chen19i.pdf) | [View Code](/models/BSP.py)
- **CDAN** - Conditional adversarial domain adaptation. [NIPS 2018](http://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation) | [View Code](/models/CDAN.py)
- **CORAL** - Deep coral: Correlation alignment for deep domain adaptation. [ECCV 2016](https://arxiv.org/abs/1607.01719) | [View Code](/models/CORAL.py)
- **CotMix** - Temporal mixup with contrastive learning and domain adversarial training. [View Code](/models/CotMix.py)
- **DAN** - Learning transferable features with deep adaptation networks. [ICML 2015](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-adaptation-networks-icml15.pdf) | [View Code](/models/DAN.py)
- **DANN** - Unsupervised domain adaptation by backpropagation. [ICML 2015](http://proceedings.mlr.press/v37/ganin15.pdf) | [View Code](/models/DANN.py)
- **DVSMN** - Dual-view style mixing network for unsupervised cross-domain fault diagnosis with imbalanced data. [Knowledge-Based Systems 2023] | [Reference Code](https://github.com/CQU-ZixuChen/DVSMN) | [View Code](/models/DVSMN.py)
- **FANDA** - Fine-grained adversarial network method for cross-domain industrial fault diagnosis. [IEEE TASE 2019] | [Reference Code](https://github.com/chunhuiz/FANDA-for-Cross-Domain-Industrial-Fault-Diagnosis) | [View Code](/models/FANDA.py)
- **MCD** - Maximum classifier discrepancy for unsupervised domain adaptation. [CVPR 2018](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf) | [View Code](/models/MCD.py)
- **MDD** - Bridging theory and algorithm for domain adaptation. [ICML 2019](http://proceedings.mlr.press/v97/zhang19i/zhang19i.pdf) | [View Code](/models/MDD.py)
- **MFSAN** - Aligning domain-specific distribution and classifier for cross-domain classification from multiple sources. [AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4551) | [View Code](/models/MFSAN.py)
- **MSI2DA_Net** - Two-stage importance-aware subgraph convolutional network based on multi-source sensors for cross-domain fault diagnosis. [Neural Networks 2024] | [Reference Code](https://github.com/Polimi-YuYue/MSI2DA-Net) | [View Code](/models/MSI2DA_Net.py)
- **MSSA** - Multi-source information transfer learning with subdomain adaptation for cross-domain fault diagnosis. [Knowledge-Based Systems 2022](https://reader.elsevier.com/reader/sd/pii/S0950705122001927) | [View Code](/models/MSSA.py)
- **OT_FaultDiagnosis** - Optimal transport (Sinkhorn divergence / Wasserstein distance) for cross-domain fault diagnosis. [Reference Code](https://github.com/Pear-so/OT_Fault-Diagnosis) | [View Code](/models/OT_FaultDiagnosis.py)
- **SLARDA** - Self-supervised autoregressive domain adaptation for time series data. [IEEE TNNLS 2022] | [Reference Code](https://github.com/mohamedr002/SLARDA) | [View Code](/models/SLARDA.py)

### Domain Generalization

- **IRM** - Invariant risk minimization. [ArXiv 2019](https://arxiv.org/abs/1907.02893) | [View Code](/models/IRM.py)
- **Mixstyle** - Domain generalization with mixstyle. [ICLR 2021](https://arxiv.org/abs/2104.02008) | [View Code](/models/Mixstyle.py)
- **IBN** - Two at once: Enhancing learning and generalization capacities via IBN-Net. [ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xingang_Pan_Two_at_Once_ECCV_2018_paper.pdf) | [View Code](/models/IBN.py)
- **MMDG** - Learning to generalize: Meta-learning for domain generalization. [AAAI 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11596) | [View Code](/models/MMDG.py)
- **DRO** - Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. [ICLR 2020](https://arxiv.org/pdf/1911.08731.pdf) | [View Code](/models/DRO.py)
- **CCDG** - Contrastive condition domain generalization for time series data, adapted to unsupervised domain adaptation. [Reference Code](https://github.com/mohamedr002/CCDG) | [View Code](/models/CCDG.py)
- **DGNIS** - Domain generalization network combining invariance and specificity for real-time fault diagnosis. [MSSP 2022] | [Reference Code](https://github.com/CHAOZHAO-1/DGNIS) | [View Code](/models/DGNIS.py)
- **DWCN_SDGFD** - Discrete wavelet convolutional network with cross-contrast perturbation for single-domain generalization. [MSSP 2025] | [Reference Code](https://github.com/JC952/DWCN-SDGFD) | [View Code](/models/DWCN_SDGFD.py)

### Baseline

- **CNN** - Plain multi-scale 1-D CNN feature extractor + MLP classifier (no domain adaptation). [View Code](/models/CNN.py)

> Note: the DG-origin methods (CCDG, DGNIS, DWCN_SDGFD) are adapted here
> to the single-source unsupervised setting used throughout this repository; they are trained with
> the same source/target pipeline as the DA methods.

## Datasets

### a) Dataset Selection

We provide loading code for **12 public fault-diagnosis datasets** (see `data_loader/load_methods.py`):

| Dataset | Description |
| --- | --- |
| [CWRU](https://engineering.case.edu/bearingdatacenter) | Case Western Reserve University bearing dataset. |
| [MFPT](https://www.mfpt.org/fault-data-sets) | Machinery Failure Prevention Technology dataset. |
| [PU](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter) | Paderborn University dataset. |
| OU | University of Ottawa bearing dataset. |
| [XJTU](https://biaowang.tech/xjtu-sy-bearing-datasets) | Xi'an Jiaotong University bearing dataset. |
| SEBEAR / SEGEAR | Southeast University gearbox dataset (bearing and gearbox subsets). |
| JNU | Jinan University bearing dataset (CSV format). |
| GB / GBN / GBR | Wind-turbine gearbox dataset (healthy vs. broken tooth; different sensor channels). |
| SQ | Bearing dataset (3 conditions, `condition_0`–`condition_2`). |

> The IMS dataset loader is provided but commented out in `data_loader/load_methods.py`.

### b) Dataset Download

For user's convenience, we also provide the pre-organized datasets. They can be manually downloaded
from:

https://pan.quark.cn/s/eb6c28ed153f

Place the downloaded datasets into a folder named `datasets` in the repository root, following the
layout below.

### c) Data Organization

For analyzing a specific dataset under different working conditions (within-dataset transfer),
divide the dataset into folders named `condition_0`, `condition_1`, etc. (one per operating
condition), each containing subfolders for the fault categories:

```
.
└── datasets
    └── CWRU
        ├── condition_0
        │   ├── ball_07
        │   │   └── 122.mat
        │   ├── inner_07
        │   │   └── 109.mat
        │   ...
        ├── condition_1
        │   ├── ball_07
        │   │   └── 123.mat
        │   ...
        ├── condition_2
        ...
```

For transfer between different datasets (cross-dataset transfer), organize the source and target so
that the fault categories correspond directly (the folder names must be consistent):

```
.
└── datasets
    ├── CWRU
    │   ├── inner
    │   ├── normal
    │   └── outer
    └── MFPT
        ├── inner
        ├── normal
        └── outer
```

> **NOTE:** the names of the folders in the source directory must be consistent with those in the
> target directory (i.e. the source and target classes must be the same).

### d) Custom Dataset Integration

To integrate other public datasets or your own data, add a loading function to
`data_loader/load_methods.py` following this template:

```python
def your_dataset_name(item_path):
    # Your code to extract the signal or data from the file
    signal = take_out_data_from_file(item_path)
    return signal
```

## Requirements

The recommended requirements are specified as follows:

* Python 3 (>=3.8)
* PyTorch (>=1.10)
* NumPy (>=1.21.2)
* Pandas (>=1.5.3)
* tqdm (>=4.46.1)
* SciPy (>=1.10)

## Run Experiments

Before running, make sure you have the desired datasets placed under `datasets/`.

The main entry point is `train.py`. The key command-line arguments are summarized below (see
`opt.py` for the full list):

| Argument | Default | Description |
| --- | --- | --- |
| `--model_name` | `CNN` | Model name (a file in `models/`); append `_ResNet` for the ResNet backbone. |
| `--source` | `CWRU_0` | Source domain(s), comma-separated (e.g. `CWRU_0,CWRU_1`). |
| `--target` | `CWRU_1` | Target domain (e.g. `CWRU_1`). |
| `--train_mode` | `single_source` | `single_source` / `source_combine` / `multi_source`. |
| `--data_dir` | `./datasets` | Directory of the datasets. |
| `--cuda_device` | `0` | GPU device id (empty string for CPU). |
| `--max_epoch` | `40` | Number of epochs. |
| `--batch_size` | `64` | Batch size. |
| `--signal_size` | `1024` | Signal length split by sliding window. |
| `--lr` | `0.01` | Initial learning rate. |
| `--opt` | `sgd` | Optimizer (`sgd` / `adam`). |
| `--load_path` | `''` | Load a checkpoint for testing (not for resuming training). |

Sample running scripts for the three training modes:

```bash
# Single-source transfer (SUDA)
python train.py --model_name DAN --source CWRU_0 --target CWRU_1 --train_mode single_source --cuda_device 0

# Source-combine transfer (multiple sources merged into one source)
python train.py --model_name DAN --source CWRU_0,CWRU_1 --target CWRU_2 --train_mode source_combine --cuda_device 0

# Multi-source transfer (MUDA)
python train.py --model_name MFSAN --source CWRU_0,CWRU_1 --target CWRU_2 --train_mode multi_source --cuda_device 0

# Use the 1-D ResNet backbone instead of the default multi-scale CNN
python train.py --model_name DAN_ResNet --source CWRU_0 --target CWRU_1 --train_mode single_source --cuda_device 0
```

Given `--model_name`, `--train_mode`, `--source`, and `--target`, checkpoints and logs are saved to:

```
ckpt/<model_name>/<train_mode>/[<sources>]To[<target>]_MMDD-HHMMSS.pth
ckpt/<model_name>/<train_mode>/[<sources>]To[<target>]_MMDD-HHMMSS.log
```

Load a trained checkpoint and evaluate on a target condition (the `--source` flag may be needed for
some models to define the model structure, but the specific sources are then unimportant):

```bash
python train.py --model_name MFSAN --load_path ./ckpt/MFSAN/multi_source/***.pth --source CWRU_0,CWRU_1 --target CWRU_3 --cuda_device 0
```

### Batch-running scripts

Instead of launching `train.py` task by task, use the batch runners to sweep all source→target
combinations across datasets and collect the best accuracy / F1 into CSV files:

| Script | Purpose |
| --- | --- |
| `run.py` | Sweep `single_source` tasks across many datasets. |
| `run_multi_source.py` | Sweep `multi_source` (leave-one-out) tasks. |
| `run_source_combine.py` | Sweep `source_combine` tasks (`leave_one_out` / `combinations`). |
| `run_sensitivity.py` | One-factor-at-a-time parameter-sensitivity study (max_epoch, batch_size, lr, opt, backbone, etc.). |

Edit the configuration block at the top of each script to select models, datasets, and hyperparameters.
Batch results are written to `results/` and logs to `logs/`.

### Model efficiency profiling

`measure_models.py` reports parameter count, FLOPs, and per-sample inference latency for each model
without training or loading real data:

```bash
py measure_models.py                                   # default: GPU (if available), 10 classes, signal length 1024
py measure_models.py --device cpu                      # measure on CPU
py measure_models.py --num_classes 10 --signal_size 1024
```

Results are printed as a table and saved to `model_metrics_summary.csv`.



## Acknowledgement

We thank the authors of the datasets and the open-source implementations referenced above for
generously sharing their data and code, which made this benchmark possible.
