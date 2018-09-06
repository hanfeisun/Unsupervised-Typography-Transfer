# Unsupervised Chinese Typography Transfer

## Demo and Details

See https://arxiv.org/abs/1802.02595

## Prerequisite

Install Python3 (>=3.5)

Put two fonts (NotoSansCJK.ttc, NotoSerifCJK.ttc) under the `./fonts` folder.

## Installation

Type the following command in shell:

```
  pip3 install --user -r requirements.txt
```

If you have a GPU that supports CUDA, replace `tensorflow` with `tensorflow-gpu` in `requirements.txt`

## Create sample from fonts

Run 

```
  make sample
```

under the root directory of the project.

## Train a model

Run

```
  python3 train.py
```

## Monitor the progress with Tensorboard

While the model is being trained, open a new tab and type

```
  tensorboard --logdir ./board
```
