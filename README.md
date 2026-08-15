## Installation

Install the environment using [uv](https://github.com/astral-sh/uv):

```shell
uv sync
```

A few examples need PyTorch. Add the extra that matches your CUDA toolkit:

```shell
uv sync --extra pt-cu126       # CUDA 12.6
uv sync --extra pt-cu118       # CUDA 11.8
uv sync --extra pt-cu128       # CUDA 12.8
uv sync --extra pt-cpu         # CPU only
```
