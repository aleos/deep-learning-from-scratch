# Deep Learning from Scratch

A neural network built from scratch in Python, following
[3Blue1Brown's Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).

## What's here

- `network.py` — `Network` class with feedforward propagation
- `test_network.py` — tests for sigmoid and network feedforward

## Setup

Requires Python 3.12+.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running tests

```
pytest -v
```

## MNIST data

Download the MNIST dataset files into the project root:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

## Progress

- [x] Chapter 1 — Network structure and feedforward
- [ ] Chapter 2 — Gradient descent (SGD)
- [ ] Chapter 3 — Backpropagation
- [ ] Chapter 4 — Backpropagation calculus
