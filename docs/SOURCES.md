# Tools and implementation references

The project uses existing maintained libraries as launch points rather than building a renderer or autodifferentiation engine. Project modules are original integration code; the linked examples are API references, not copied implementations.

| Resource | Use |
| --- | --- |
| [Bullet / PyBullet](https://github.com/bulletphysics/bullet3) | 3D geometry, CPU TinyRenderer, camera matrices, and independent simulation clients |
| [Official PyBullet camera example](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/testrender_np.py) | Camera-image API reference; this project selects the CPU renderer for headless execution |
| [PyTorch](https://pytorch.org/) | Small CNN, contrastive learning, checkpointing, CPU inference |
| [Official PyTorch optimization tutorial](https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/optimization_tutorial.py) | Training/evaluation API patterns |
| [Official PyTorch MNIST example](https://github.com/pytorch/examples/blob/main/mnist/main.py) | Reference for a compact, reproducible training workflow |
| [uv](https://docs.astral.sh/uv/) | Python 3.11 environment management and locked dependency installation |
| [pytest](https://docs.pytest.org/) | Functional and regression tests |

PyBullet 3.2.7 is pinned and Python 3.11 is selected because an official Linux x86-64 wheel is available for that interpreter. PyTorch is resolved from its official CPU wheel index, avoiding unnecessary CUDA packages. `uv.lock` records the concrete dependencies and artifact hashes. The first baseline uses procedural geometry, so no external scene assets or asset downloads are needed at runtime.

The ant-navigation papers listed in the original project proposal remain literature-review leads. This implementation does not claim to reproduce those papers, establish a new research gap, or validate a biological mechanism.
