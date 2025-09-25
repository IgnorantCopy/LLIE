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

### Train a Model

```shell
export PYTHONPATH=$(pwd)
cd src/llie
pixi run --environment llie-gpu python train.py --config ./configs/method/config.yaml
```

### Test a Model

```shell
export PYTHONPATH=$(pwd)
cd src/llie
pixi run --environment llie-gpu python test.py  --ckpt path/to/checkpoint.ckpt --config ./configs/test/config.yaml
```