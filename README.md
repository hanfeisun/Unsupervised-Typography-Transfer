# Unsupervised Chinese Typography Transfer

## Demo and Details

See https://zhuanlan.zhihu.com/p/31619824

Or https://arxiv.org/abs/1802.02595 for details

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

## Acknowledgements
Code derived from:

* [zi2zi](https://github.com/kaonashi-tyc/zi2zi) by [Yuchen Tian](https://github.com/kaonashi-tyc)

Network architecture derived from:

* [Chinese typography transfer](https://arxiv.org/abs/1707.04904) by Jie Chang and Yujun Gu

## License
Apache 2.0
