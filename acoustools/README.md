# AcousTools

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/acoustools.svg)](https://pypi.org/project/acoustools)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/acoustools.svg)](https://pypi.org/project/acoustools) -->

A Full-Stack Python based library for working with acoustic fields for holgraphy. Developed using PyTorch, AcousTools uses PyTorch Tensors to represent points, acoustic fields and holograms to enable development of new algorithms, applications and acoustic systems. As a full-stack solution, Acoustools is able to implement each stage of development making it a single point of call.

See [Here](https://github.com/JoshuaMukherjee/AcousticExperiments/tree/main/AcousTools_Examples) for examples of code using AcousTools.
The [Preprint of AcousTools can be found on arXiv](https://arxiv.org/abs/2511.07336)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)
- [Docs](#documentation)

## Installation


Optionally create a virtual environment to install AcousTools into <br>
Optionally install the correct version of [PyTorch](https://pytorch.org/get-started/locally/) <br>

Run 
```console
pip install acoustools
```
Or visit [AcousTools' on PyPi](https://pypi.org/project/acoustools/)

### Local Installation

Clone the repo and then run

```console
pip install -r <path-to-clone>/requirements.txt
pip install -e <path-to-clone>/acoustools/ --config-settings editable_mode=strict
```

Use `python<version> -m` before the above commands to use a specific version of python.

where `<path-to-clone>` is the local location of the repository 

## Documentation

#### Documentation can be seen [Here](https://joshuamukherjee.github.io/AcousTools/src/acoustools.html)

Or to view the documentation for AcousTools locally, firstly install pdoc:
```console
pip install pdoc
```
Then run pdoc on AcousTools to create a locally hosted server containing the documentation
```console
python -m pdoc <path-to-clone>/acoustools/ --math
```

See [Here](https://github.com/JoshuaMukherjee/AcousticExperiments/tree/main/AcousTools_Examples) for examples of code using AcousTools.


## AcousTools Basics

AcousTools represents data as `torch.Tensors`. A point is represented as a tensor where each column represents a (x,y,z) point. Groups of points can also be grouped into batches of points for parallel computation and so have a shape (B,3,N) for B batches and N points.

Ultrasound waves can be focused by controlling many sources such that at a given point in space all waves arrive in phase and therefore constructivly interfere. This can be done in a number of ways (`acoustools.Solvers`). This allows for applications from high speed persistance-of-vision displays to haptic feedback and non-contact fabrication. 

## License

`acoustools` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
