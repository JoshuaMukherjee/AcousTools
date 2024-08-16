# AcousTools

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/acoustools.svg)](https://pypi.org/project/acoustools)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/acoustools.svg)](https://pypi.org/project/acoustools) -->

A Python based library for working with acoustic fields for levitation. Developed mostly using PyTorch, AcousTools uses PyTorch Tensors to represent points, acoustic fields and holograms to enable development of new algorithms, applications and acoustic systems. 

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

Clone the repo and then run

```console
pip install -r <path-to-clone>/requirements.txt
pip install -e <path-to-clone>/acoustools/ --config-settings editable_mode=strict
```

where `<path-to-clone>` is the local location of the repository 

## AcousTools Basics

AcousTools represents data as `torch.Tensors`. A point is represented as a tensor where each column represents a (x,y,z) point. Groups of points can also be grouped into batches of points for parallel computation and so have a shape (B,3,N) for B batches and N points.

Ultrasound waves can be focused by controlling many sources such that at a given point in space all waves arrive in phase and therefore constructivly interfere. This can be done in a number of ways (`acoustools.Solvers`). This allows for applications from high speed persistance-of-vision displays to haptic feedback and non-contact fabrication. 

## License

`acoustools` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
