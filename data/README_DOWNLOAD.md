# Downloading the Datasets

These datasets are hosted on Google Drive / university servers, not on GitHub,
so you need to fetch them yourself.

## LEVIR-CD (recommended)

Official page: https://justchenhao.github.io/LEVIR/
- Contains a Google Drive link to `LEVIR-CD.zip` (~1.9 GB), already split into
  train/val/test with `A/`, `B/`, `label/` subfolders at 1024x1024.
- Also mirrored on Kaggle — search "LEVIR-CD" on kaggle.com and use
  `kaggle datasets download` if you have a Kaggle API token configured.

Once downloaded, unzip so you get:
```
dataset/train/A dataset/train/B dataset/train/label
dataset/val/A   dataset/val/B   dataset/val/label
dataset/test/A  dataset/test/B  dataset/test/label
```
Then run:
```
python data/prepare_dataset.py --patchify --src dataset --dst dataset_patched --patch_size 256
```
This cuts the native 1024x1024 images into 256x256 patches (16 patches per
image), which trains much faster and is standard practice for LEVIR-CD.

## WHU-CD

Page: http://gpcv.whu.edu.cn/data/building_dataset.html
Comes as two giant orthophotos (2012 / 2016) + a change map, which you must
tile yourself. `data/prepare_dataset.py --patchify` handles arbitrary large
images too — just point `--src` at a folder with `A/2012.tif`, `B/2016.tif`,
`label/change.tif` and add `--single_pair` mode (see script docstring).

## DSIFN-CD

Page: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-optical-remote-sensing-images
Already 512x512 patches with train/val/test split matching the LEVIR-CD
folder convention — drop straight in.

## No internet / waiting on download? Use synthetic data now

```
python data/prepare_dataset.py --synthetic --out dataset_synth --n_train 400 --n_val 80 --n_test 80
```

This procedurally generates plausible-looking before/after satellite-style
patches (textured land-cover + randomly inserted "buildings"/"roads" that
appear/disappear between the two timestamps) with matching ground-truth change
masks, in the exact LEVIR-CD folder format. It lets you build, debug and even
partially train the whole pipeline today while the real dataset downloads in
the background. Swap `--data_root dataset_synth` for `--data_root
dataset_patched` in the config once real data is ready — nothing else changes.
