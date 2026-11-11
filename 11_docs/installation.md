# Installation

Use Linux, an NVIDIA GPU, and a CUDA-compatible PyTorch installation. The paper
environment used Python 3.12, PyTorch 2.5.1 with CUDA 12.4, and Triton 3.1.0.
Install the appropriate PyTorch CUDA wheel for your system before installing
LEAP if the default package index does not supply it.

```bash
python -m pip install -r 10_requirements/reproduction.txt
python -m pip install -e . --no-deps
python -m leap check
python -m leap checkpoints
```

`check` verifies checkpoints and configuration files and reports missing data.
After preparing the [datasets](../04_data/README.md), use `--require-data` to make
missing inputs an error. External baselines have their own author dependencies
and should run in separate environments. See [external models](../07_external_models/external_models/README.md).

Continue with the [quick start](quick_start.md) or return to the [overview](../README.md).
