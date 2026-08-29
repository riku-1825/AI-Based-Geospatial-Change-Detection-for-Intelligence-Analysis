[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](#license)

# AI-Based Geospatial Change Detection for Intelligence Analysis

## Overview

This repository implements a **bi-temporal satellite image change detection pipeline** for geospatial intelligence analysis. Given two co-registered satellite images of the same geographic area captured at different times (T1 = "before", T2 = "after"), the model automatically localizes regions of significant surface change — new construction, road/infrastructure changes, land-use conversion, vegetation change, and large structural changes — and produces a pixel-wise **change mask** along with a **quantitative change summary** (e.g. `Changed Area: 14.7% | Unchanged Area: 85.3%`).

The task is framed as **generic binary change detection** (changed vs. unchanged), the standard formulation used across remote-sensing benchmarks such as LEVIR-CD, WHU-CD, and DSIFN-CD — it identifies *where* meaningful surface change occurred, not *what* changed.

The core model is a **Siamese U-Net** with a shared, ImageNet-pretrained **ResNet (18/34)** encoder: both temporal images pass through the same weight-shared encoder, multi-scale features are differenced at each resolution, and a U-Net-style decoder reconstructs a full-resolution change probability map. This gives a strong accuracy/compute tradeoff compared to heavier transformer-based approaches (e.g. ChangeFormer) while still converging quickly thanks to pretrained weights.

---

## Sample Results

<p align="center">
  <img src="https://github.com/riku-1825/AI-Based-Geospatial-Change-Detection-for-Intelligence-Analysis/blob/main/images/synthetic.png" width="80%" />
</p>

<p align="center">
<b>Fig 1:</b> Synthetic dataset (before / after / predicted change overlay)
</p>

<p align="center">
  <img src="https://github.com/riku-1825/AI-Based-Geospatial-Change-Detection-for-Intelligence-Analysis/blob/main/images/LEVIR_CD.png" width="80%" />
</p>

<p align="center">
<b>Fig 2:</b> LEVIR-CD real satellite data (before / after / predicted change overlay)
</p>

---

## Features

- **Siamese U-Net** architecture with a shared, ImageNet-pretrained ResNet18/ResNet34 encoder
- Multi-scale **feature-differencing** fusion (standard change-detection strategy, à la FC-Siam-diff)
- **BCE + Dice** combo loss to handle the heavy class imbalance typical of change masks
- Mixed-precision (AMP) training with checkpointing and TensorBoard logging
- Works out-of-the-box with the **LEVIR-CD** benchmark, and includes a **synthetic data generator** to validate the full pipeline without waiting on a dataset download
- Automatic **patchifying** of native 1024×1024 LEVIR-CD imagery into 256×256 training patches
- **Folder-level inference**: run change detection over an entire folder of before/after image pairs in one command, with per-image visualizations, binary masks, and a summary CSV
- Quantitative evaluation: Precision, Recall, F1, IoU, Overall Accuracy, and changed/unchanged area percentages

---

## Repository Structure

```
├── data/                      # Dataset loading & preparation
│   ├── dataset.py             # PyTorch Dataset for LEVIR-CD-format data
│   └── prepare_dataset.py     # Synthetic data generator + patchify utility
├── models/
│   └── siamese_unet.py        # Siamese U-Net (ResNet encoder) model
├── utils/
│   ├── losses.py              # BCE + Dice combo loss
│   ├── metrics.py             # Precision/Recall/F1/IoU/OA + area stats
│   └── visualize.py           # Result visualization helpers
├── configs/
│   └── siamese_unet
│       └── config.yaml         # All hyperparameters in one place           
├── checkpoints/                # Saved model checkpoints (best.pt, last.pt)
├── outputs/                    # Evaluation & inference result images
├── runs/                       # TensorBoard logs
├── train.py                    # Training loop (AMP, checkpointing, logging)
├── evaluate.py                 # Test-set evaluation
├── inference.py                # Folder-level before/after inference
├── environment.yaml            # Conda environment definition
├── requirements.txt            # Pip dependencies
├── LICENSE
└── README.md
```

---

## Dataset

This project is trained and evaluated on **LEVIR-CD**, the most widely used building-change-detection benchmark, consisting of bi-temporal Google Earth image pairs at 0.5m/pixel resolution with binary building-change masks.

### Dataset Statistics

| Split | Image Pairs (native 1024×1024) | Patches (256×256, after patchify) |
| ----- | ------------------------------- | ---------------------------------- |
| Train | 445                              | 7,120                              |
| Val   | 64                               | 1,024                               |
| Test  | 128                              | 2,048                               |
| **Total** | **637**                     | **10,192**                          |

### Dataset Links

| Source | Link |
| ------------------- | ------------------------------------------------------- |
| Official LEVIR-CD page | <https://justchenhao.github.io/LEVIR/> |
| OneDrive — raw LEVIR-CD (native 1024×1024) | <https://1drv.ms/f/c/279f2820d299d227/IgDASApj2oKUTL5A3Wx9v7hgAWJLN_8Hj78VlWdjo2EGC70?e=ZYhmqx> |
| OneDrive — patched LEVIR-CD (256×256, ready to train) | <https://1drv.ms/f/c/279f2820d299d227/IgBj0rgk7m7CTrzCdFNxBTTeAfpDJtmuxNVHoQEAHFkpM8Y?e=TWEcB6> |

---

## Data Preparation

### Patchifying the raw LEVIR-CD data

The native LEVIR-CD images are 1024×1024, which is larger than necessary for efficient training. We split each image into **256×256 patches** (16 patches per image), which is standard practice on this dataset — it increases the effective number of training samples and keeps memory/compute requirements low without losing spatial detail (0.5m/pixel resolution is preserved per patch).

```bash
python data/prepare_dataset.py --patchify --src LEVIR_CD_Dataset --dst dataset_patched --patch_size 256
```

This expects `LEVIR_CD_Dataset/{train,val,test}/{A,B,label}` (the native LEVIR-CD folder layout) and writes the patched dataset to `dataset_patched/{train,val,test}/{A,B,label}`. Once done, point `data_root` in `configs/siamese_unet/config.yaml` to `dataset_patched`.

### Generating synthetic data

To validate the entire pipeline (data loading, training loop, losses, metrics, visualization) without waiting on the real dataset download, a synthetic before/after/change-mask generator is included:

```bash
python data/prepare_dataset.py --synthetic --out dataset_synthetic --n_train 400 --n_val 80 --n_test 80 --img_size 256
```

This procedurally generates plausible before/after satellite-style image pairs with matching ground-truth change masks, in the exact LEVIR-CD folder format, so it's a drop-in swap for `data_root` in the config.

---

## Setup

Clone the repository and create the Conda environment from the YAML file:

```bash
git clone https://github.com/riku-1825/AI-Based-Geospatial-Change-Detection-for-Intelligence-Analysis.git
cd AI-Based-Geospatial-Change-Detection-for-Intelligence-Analysis

conda env create -f environment.yaml
conda activate sat_img
```

---

## Model Checkpoints

Trained checkpoints (Siamese U-Net with ResNet18 and ResNet34 backbones, on both LEVIR-CD and synthetic data) are available via the OneDrive link below.

[![Download](https://img.shields.io/badge/Download-Model%20Checkpoints-0078D4?style=for-the-badge&logo=microsoftonedrive&logoColor=white)](https://1drv.ms/f/c/279f2820d299d227/IgCKCatFhFM-TqZvs5OlG8TaAY2gmE9XxdGRX0oTRZJgWdo?e=zFwA2p)

---

## How to Run

### Training

```bash
python train.py --config configs/siamese_unet/config.yaml
```
Trains the Siamese U-Net on the dataset specified by `data_root` in `configs/config.yaml`, saving the best checkpoint (by validation F1) to `checkpoints/best.pt` and TensorBoard logs to `runs/`.

### Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best.pt --config configs/siamese_unet/config.yaml --n_qualitative 6 --out_dir outputs/eval
```
Runs the trained model on the test split, reporting Precision, Recall, F1, IoU, Overall Accuracy, and mean changed/unchanged area percentage, and saves qualitative before/after/GT/predicted figures to `outputs/eval`.

### Inference (folder of before/after image pairs)

```bash
python inference.py --before path/to/before_folder --after path/to/after_folder \
    --checkpoint checkpoints/best.pt --config configs/siamese_unet/config.yaml \
    --out outputs/inference_results
```
Matches before/after images by filename across the two folders, and for every pair writes a comparison figure and a binary change mask to `--out`, plus a single `summary.csv` covering changed/unchanged area statistics for the whole folder.

---

## Results

| Dataset | Architecture | Backbone | Precision | Recall | F1 Score | IoU | Overall Accuracy | GT Changed Area (%) | Mean Predicted Changed Area (%) |
| ------- | ------------ | -------- | --------- | ------ | -------- | --- | ----------------- | -------------------- | --------------------------------- |
| LEVIR-CD | Siamese U-Net | ResNet18 | 0.8784 | 0.8997 | 0.8889 | 0.8001 | 0.9885 | 5.09 | 5.22 |
| LEVIR-CD | Siamese U-Net | ResNet34 | 0.8858 | 0.8937 | 0.8897 | 0.8014 | 0.9887 | 5.09 | 5.14 |
| Synthetic | Siamese U-Net | ResNet18 | 0.9944 | 0.9944 | 0.9944 | 0.9889 | 0.9993 | 5.56 | 5.56 |
| Synthetic | Siamese U-Net | ResNet34 | 0.9942 | 0.9944 | 0.9932 | 0.9887 | 0.9993 | 5.56 | 5.56 |

### Key Observations

- On real LEVIR-CD data, the deeper **ResNet34** backbone gives a small but consistent edge over ResNet18 (F1 0.8897 vs. 0.8889, IoU 0.8014 vs. 0.8001), while on the synthetic dataset the simpler **ResNet18** backbone is marginally better (F1 0.9944 vs. 0.9932) — the synthetic task is easy enough that extra encoder capacity yields no benefit and the metrics are already near-saturated.
- Predicted changed-area percentage tracks ground-truth changed-area percentage closely on LEVIR-CD (5.22% predicted vs. 5.09% GT for ResNet18; 5.14% vs. 5.09% for ResNet34), indicating the model's area estimates — the actual quantity of interest for intelligence-style change summaries — are well calibrated even where pixel-level F1/IoU are imperfect.
- There is a substantial performance gap between synthetic data (F1 ≈ 0.99) and real LEVIR-CD data (F1 ≈ 0.89), reflecting the real-world **domain gap**: real imagery includes sensor noise, illumination/seasonal differences, and imperfect co-registration that the procedurally generated synthetic data does not fully capture. This confirms synthetic data is well suited for pipeline validation but not a substitute for real benchmark evaluation.

### Future Work

- **Attention-based / transformer fusion**: incorporate spatial-temporal attention (e.g. STANet) or a transformer-based encoder-decoder (e.g. ChangeFormer) to better model long-range context and close the real-data domain gap observed above.
- **Multi-class change-type classification**: extend the current binary changed/unchanged formulation to classify the *type* of change (e.g. new construction vs. vegetation loss vs. road change), which would require multi-class annotated data such as DSIFN-CD.

---

## References

This project builds on the following methods and datasets:

### LEVIR-CD Dataset / STANet

- **Paper**
Chen, H., & Shi, Z. (2020). *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*. Remote Sensing, 12(10), 1662.

```bibtex
@article{chen2020spatial,
  title={A spatial-temporal attention-based method and a new dataset for remote sensing image change detection},
  author={Chen, Hao and Shi, Zhenwei},
  journal={Remote Sensing},
  volume={12},
  number={10},
  pages={1662},
  year={2020},
  publisher={MDPI}
}
```

### Fully Convolutional Siamese Networks for Change Detection

- **Paper**
Daudt, R. C., Le Saux, B., & Boulch, A. (2018). *Fully Convolutional Siamese Networks for Change Detection*. IEEE ICIP.

```bibtex
@inproceedings{daudt2018fully,
  title={Fully convolutional siamese networks for change detection},
  author={Daudt, Rodrigo Caye and Le Saux, Bertrand and Boulch, Alexandre},
  booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
  pages={4063--4067},
  year={2018},
  organization={IEEE}
}
```

### ChangeFormer

- **Paper**
Bandara, W. G. C., & Patel, V. M. (2022). *A Transformer-Based Siamese Network for Change Detection*. IEEE IGARSS.

```bibtex
@inproceedings{bandara2022transformer,
  title={A transformer-based siamese network for change detection},
  author={Bandara, Wele Gedara Chaminda and Patel, Vishal M},
  booktitle={IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium},
  pages={207--210},
  year={2022},
  organization={IEEE}
}
```

### U-Net

- **Paper**
Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.

```bibtex
@inproceedings{ronneberger2015unet,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

### ResNet

- **Paper**
He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. IEEE CVPR.

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={770--778},
  year={2016}
}
```

---

## Citation

If you find this repository useful in your research, please consider citing it:

```bibtex
@misc{bagh2026geospatialchangedetection,
  author       = {Bhoumik Chandra Bagh},
  title        = {AI-Based Geospatial Change Detection for Intelligence Analysis},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/riku-1825/AI-Based-Geospatial-Change-Detection-for-Intelligence-Analysis}},
  note         = {GitHub repository}
}
```

---

## License

This project is licensed under the **MIT License**.

You are free to:

- Use the source code for research and educational purposes.
- Modify and distribute the code under the terms of the MIT License.
- Include this work in your own projects with appropriate attribution.

For more details, see the [LICENSE](LICENSE) file.

