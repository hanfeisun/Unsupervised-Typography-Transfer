# Unsupervised Chinese Typography Transfer

## Prerequisite

Install Python3 (>=3.5)

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
