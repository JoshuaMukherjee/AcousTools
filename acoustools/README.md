# AcousTools

AcousTools is an acoustic holography toolkit which enables easy development of applications from start to finish. The idea is to allow developers to use AcousTools from idea generation to simulation to visualisation to deployment on hardware. The Acoustic Full-stack explains the stages of creating an acoustic application and AcousTools can be used at each step:

- Setup: Locations of points to focus ultrasound at, the arrays to be used and definition of any paths that the focus may move along
- Propagators: How simulation of the waves from array -> target point will be computed
- Solver: Given a propagator, compute the signals that the array would need to focus sound at the points defined
- Analysis: Compute and potentially visualise the field using various metrics
- Hardware: Send the signals to real hardware and create real-world ultrsound devices.

AcousTools is based on PyTorch and uses tensors to represent the data. This means that CUDA based GPU acceleration is available if your computer has a CUDA GPU. The shapes of these tensors are:

- Points $z \in \mathbb{C}^{B \times 3 \times Z}$
- Hologram $x \in \mathbb{C}^{B \times T \times 1}$
- Array $\tau \in \mathbb{C}^{T \times 3}$
- Propagator $A \in \mathbb{C}^{B \times Z \times T}$

This tutorial will aim to help a new developer to get started with AcousTools from which point they can explore the documentation further.

See [Here](https://github.com/JoshuaMukherjee/AcousticExperiments/tree/main/AcousTools_Examples) for examples of code using AcousTools.
The [Paper for AcousTools can be seen Here](https://ieeexplore.ieee.org/document/11369395)
The [Preprint of AcousTools can be found on arXiv](https://arxiv.org/abs/2511.07336)

## Installation


Optionally create a virtual environment to install AcousTools in 
Optionally install the correct version of [PyTorch](https://pytorch.org/get-started/locally/) 

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

## Standard Interface

The standard interface to AcousTools is composed of a series of functions each of which can be composed to obtain the behaviour that the user defines. Here, we will step through a simple example in order to understand AcousTools fundamentals

### Setup
Firstly, the points and array should be defined before anything else can be done. We will use two array, pointing at one another and a point at the mid point of these two arrays as the target. AcousTools considers this central point to be the origin of the coordinate space.

```python
from acoustools.Utilities import create_point, transducers

board = transducers(16) #create two boards of 16x16 transducers

p = create_point(N=1, x=0, y=0, z=0)
```
There are built in variables for common transducer arrays - the top-bottom array, a top only array or a bottom only array. These can be defined as

```python
from acoutsools.Utilities import TOP_BOARD #16x16 transducer array above the working area
from acoutsools.Utilities import BOTTOM_BOARD #16x16 transducer array below the working area
from acoutsools.Utilities import TRANSDUCERS #Two 16x16 transducer arrays above and below the working area
```
And multiple points could be defined at once by supplying multiple values to `(x,y,z)` parameters in `create_point`


### Propagators

In order to model the sound propagation from the array to a point, a propagator is needed. This is a matrix that models how each source contributes to each of the points and is often modelled as a piston source. This can be compued using

```python
from acoustools.Utilities import forward_model

F = forward_model(points = p, transducers = board)
```

Which will model sound from `board` to `p`. This could be replaced with any other matrix such as one that consider sound scattering using BEM (see `acoustools.BEM`)

### Solver

The next stage is to compute the signals that will focus the sound waves at the points after the waves are modelled using the propagator. There are a variety of solvers which are good for various different applications. 

- `wgs`: Good for multipoint focusing with very low variance of the pressure between points. Slower computation
- `gspat`: Good for very fast computation for multipoint focusing. Larger variance in pressure but normally not too large
- `naive`: Very fast computation but very large variance in pressure.
- `kd_solver`: Very fast computation but only focuses at a single point.

For example

```python
from acoustools.Solvers import naive

x = naive(points=p, board=board, A=F)
```

But equally any other solver could be swapped in for different purposes.

### Analysis

Once a hologram (`x`) has been computed - the sound field can be validated using various different metrics

- complex pressure, $p$: The complex value of the field at a point from computing `A@x` where `@` is matric multiplication
- pressure, $|p|$ : The absolute value of the complex pressure - denotes the phsycial pressure at a point
- phase, $\varphi$ : The angle of the complex pressure - the phase of the acoustic wave
- Gor’kov pressure, $U$ : a metric of the field such that the acoustic force on a small particle is equal to $-\nabla U$
- Acoustic Radiation Force, $F_{x,y,z}$: Normally computed from the gradient of $U$ but can be computed other ways (more complex and out of scope to explain here!)

There are functions for all of these metrics
```python
from acoustools.Utilities import propagate, propagate_abs, propagate_phase
from acoustools.Gorkov import gorkov
from acoustools.Force import compute_force

p = propagate(activations=x, points=p, board=board, A=F) #p = F@x
pressure = propagate_abs(activations=x, points=p, board=board, A=F) #pressure = p.abs()
phase = propagate_phase(activations=x, points=p, board=board, A=F) #pressure = p.angle()

U = gorkov(activations=x, points=p, board=board)

fx,fy,fz = compute_force(activations=x, points=p, board=board)

```

These can also be used to visualise the whole sound field using AcousTools built in visualiser - the visualiser has a large scope for customisation and so the full explanation cannot be given here but it requires

1. The plane to be visualised - defined by three points A, B and C. A is the top left corner, B is the top right and C the bottom left. These can be custom or defined using the built in function `ABC`
2. The hologram to visualise
3. The resolution of the images to create
4. The function used to colour each pixel in the frame. eg using `propagate_abs` will render the pressure field while `gorkov` will show the Gor’kov potential at each point. A list of functions can be passed to render multipple fields
5. Any parameters that need to be passed to the functions as a list of dictionaries. The order will correspond to the order of colour functions

```python
from acoustools.Visualiser import Visualise, ABC


Visualise(*ABC(0.01),
          x, points=p, 
          colour_functions=[propagate_abs,gorkov], 
          colour_function_args=[{'board':board}, {'board':board}]
          res=(200,200))
```

### Hardware

Finally, the signals can be sent to a hardware device. Currently AcousTools uses OpenMPD’s driver and requires Windows to run. In order to send signals, simply connect and send the hologram

```python
with LevitatorController(ids = 1000,999)) as lev:
    lev.levitate(x)
```
Different arrays may have different ids to connect - simply change the id to the correct value.


### Common Issue

- Using one array shape for one function and a different one elsewhere - ensure you use the correct value across each function call
- Incorrect solver choice - make sure the solver you use is appropriate for the task (if in doubt use gspat or wgs)
  
#### Note on default values

Most functions parameters in AcousTools are optional meaning if they are not provided a common value will be assumed. This can cause problems if the assumed value is not the correct one. See the documentation for details as to what the defauls parameters are. For example

```python
p = propagate(activations=x, points=p, board=board, A=F) 
p = propagate(x,p) 
```
Are equivalent for a top-bottom array as the `TRANSDUCERS` variable is the default board that is used. The propagator `A` is also inferred from the board and the points - without explicit definition the piston model is used.


## High Level Interface

AcousTools contains an alternative very high-level interface which abstracts almost all the acoustics details away. This may or may not be the correct choice for any given user and depends how much control over the signals and solvers that you need. These are called Contexts and can be uses as:
```python
from acoustools.HighLevel import TopBottomContext

with TopBottomContext() as ctx:
    ctx.create_focus()
    ctx.send_solution()

```
This sets the context up, create a focus and send it to the physical device connected over USB. In order to change the position a `location` parameter can be applied to the `create_focus` method which should be an AcousTools point or an iterable of point locations. 

## License

`acoustools` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Citation

Please Cite AcousTools as:

```
@ARTICLE{11369395,
  author={Mukherjee, Joshua and Christopoulos, Giorgos and Shen, Zhouyang and Subramanian, Sriram and Hirayama, Ryuji},
  journal={IEEE Transactions on Ultrasonics}, 
  title={AcousTools: A “Full-Stack,” Python-Based, Acoustic Holography Library}, 
  year={2026},
  volume={73},
  number={2},
  pages={99-111},
  keywords={Acoustics;Transducers;Holography;Scattering;Levitation;Computational modeling;Three-dimensional printing;Software libraries;Acoustic applications;Python;Acoustic applications;acoustic field;levitation;software libraries},
  doi={10.1109/TUSON.2026.3659798}}
```
