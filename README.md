# LLIE
A repository for Low-Light Image Enhancement codes.

## Usage

### Install Requirements

1. install pixi (on linux)

```shell
curl -fsSL https://pixi.sh/install.sh | sh
```
> Please refer to https://pixi.sh/latest/installation/ for more information.

2. clone the repo

```shell
git clone https://github.com/IgnorantCopy/LLIE.git
cd LLIE
```

3. install dependencies

```shell
pixi install --environment llie-gpu
```
> If you do not have a gpu on your machine, please use `llie-cpu` instead of `llie-gpu`.

### Download Datasets

The original links of the datasets are provided: 
[LOLv1](https://drive.google.com/file/d/157bjO1_cFuSd0HWDUuAmcHRJDVyWpOxB), 
[LOLv2](https://drive.google.com/file/d/1dzuLCk9_gE2bFF222n3-7GVUlSVHpMYC),
[DICM, LIME, MEF, NPE, VV](https://drive.google.com/file/d/1OvHuzPBZRBMDWV5AKI-TtIxPCYY8EW70), 
[LOLBlur](https://drive.google.com/drive/folders/11HcsiHNvM7JUlbuHIniREdQ2peDUhtwX),
[SICE](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBb1BSSm1pRDI0VXBoQWxhVElla2RNTHdMWm5BP2U9V3hyZk9h&cid=2985DB836826D183&id=2985DB836826D183%21521&parId=2985DB836826D183%21384&o=OneUp) (code: `yixu`),
[SID](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBb1BSSm1pRDI0VXBoQWllOWwwRHVNTjIwUEI3P2U9WmM1RGNB&cid=2985DB836826D183&id=2985DB836826D183%21520&parId=2985DB836826D183%21384&o=OneUp) (code: `yixu`),
[EnlightenGAN dataset](https://drive.google.com/drive/folders/1KivxOm79VidSJnJrMV9osr751UD68pCu),
[ZeroDCE dataset](https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3),

You can put the downloaded datasets in the same folder, and run the script to replace all the data root in config files:

```shell
export PYTHONPATH=$(pwd)
cd src/llie
pixi run --environment llie-gpu python replace_data_root.py --root /path/to/dataroot
```

### Train a Model

```shell
pixi run --environment llie-gpu python train.py --config ./configs/method/config.yaml
```

### Test a Model

```shell
pixi run --environment llie-gpu python test.py  --ckpt path/to/checkpoint.ckpt --config ./configs/test/config.yaml
```