# Box2D installation on Python 3.14

## Purpose

This document records how `box2d-py` was installed for Gymnasium's
`LunarLander-v3` environment. It preserves both failed attempts, their root
causes, and the successful recovery procedure so the environment can be
reproduced or debugged later.

## Environment

- Date: 2026-10-25
- Operating system: Linux x86_64
- Virtual environment: `/data/AI_Development/reinforcement-learning/.venv`
- Python: 3.14.4
- pip: 26.2.2
- setuptools: 78.1.0
- wheel: 0.48.0
- packaging: 26.3
- SWIG: 4.5.1
- Requested Box2D package: `box2d-py==2.3.8`
- Requested Gymnasium extra: `gymnasium[box2d]`

`box2d-py` contains a native C++ extension. Unlike a pure Python package, it
must generate binding code and compile that code for the active Python version
and operating system.

## Attempt 1: SWIG executable was unavailable

Command:

```bash
python -m pip install "gymnasium[box2d]"
```

Important error:

```text
error: command 'swig' failed: No such file or directory
ERROR: Failed building wheel for box2d-py
```

### Cause

The `box2d-py` source build runs SWIG to generate Python/C++ binding code.
Gymnasium's `box2d` extra caused pip to resolve `swig==4.*`, but pip attempted
to build the Box2D wheel before the resolved SWIG package had been installed
in the reusable virtual environment.

In other words, SWIG was part of the overall resolved installation but was not
available at the exact moment the Box2D build invoked it. The failed pip
transaction was rolled back, leaving no partial Gymnasium installation.

### Action taken

SWIG was installed explicitly first:

```bash
python -m pip install swig==4.5.1
```

It was verified with:

```bash
which swig
swig -version
```

The launcher was installed at:

```text
/data/AI_Development/reinforcement-learning/.venv/bin/swig
```

## Attempt 2: build isolation hid SWIG's Python module

After installing SWIG, the original command was tried again:

```bash
python -m pip install "gymnasium[box2d]"
```

Important error:

```text
File "/data/AI_Development/reinforcement-learning/.venv/bin/swig", line 3
    from swig import swig
ModuleNotFoundError: No module named 'swig'
error: command 'swig' failed with exit code 1
```

### Why the executable was found but its module was missing

The pip-distributed SWIG executable is a Python launcher. Its relevant lines
are equivalent to:

```python
from swig import swig
```

The `swig` Python module was correctly installed in the reusable virtual
environment, and this direct test succeeded:

```bash
python -c "import swig; print(swig.__file__)"
```

However, pip used an isolated PEP 517 build environment for `box2d-py`.
Isolation prevents undeclared packages from the reusable environment's
`site-packages` directory from being imported during a build. The build found
the outer environment's `swig` launcher through `PATH`, but the launcher could
not import its companion module from the hidden outer `site-packages`.

This was therefore an interaction between:

1. A Python-based SWIG launcher installed in the reusable environment.
2. A Box2D source distribution that invokes SWIG while building.
3. Pip's isolated build environment hiding the launcher's Python module.

The failed transaction was again rolled back without leaving partially
installed Gymnasium packages.

## Successful installation

The reusable environment first received the wheel build tool:

```bash
python -m pip install wheel
```

This installed:

```text
wheel==0.48.0
packaging==26.3
```

Box2D was then built separately while allowing the build to use the reusable
environment's installed setuptools, wheel, and SWIG packages:

```bash
python -m pip install box2d-py==2.3.8 --no-build-isolation
```

Successful build artifact:

```text
Filename: box2d_py-2.3.8-cp314-cp314-linux_x86_64.whl
Size: 2,394,578 bytes
SHA-256: e0e805bbe05a04da174ef213c45f26ef323eb5cc5315971bd2fcd0cf7e775933
```

The tags in the filename have useful meanings:

- `cp314`: built for CPython 3.14.
- The second `cp314`: uses the CPython 3.14 ABI.
- `linux_x86_64`: built for 64-bit Linux on x86 hardware.

This wheel is platform-specific. It should not be assumed to work with a
different Python version, operating system, or CPU architecture.

## Verification

The installed extension was tested with:

```bash
python -c "import Box2D; from Box2D import b2World; world=b2World(gravity=(0,-10)); print(Box2D.__version__); print(tuple(world.gravity))"
```

Observed result:

```text
2.3.8
(0.0, -10.0)
```

This verifies more than package metadata: Python imported the compiled native
extension and successfully created a Box2D physics-world object.

## Reproduction procedure

For a new virtual environment using the same platform and Python version:

```bash
python -m pip install swig==4.5.1
python -m pip install wheel==0.48.0
python -m pip install box2d-py==2.3.8 --no-build-isolation
python -m pip install "gymnasium[box2d]"
```

The final Gymnasium command should detect that `box2d-py` and SWIG are already
satisfied, then install only the remaining Gymnasium and rendering packages.

## Final Gymnasium installation result

The final command completed successfully after the separate Box2D build:

```bash
python -m pip install "gymnasium[box2d]"
```

Pip reused `box2d-py==2.3.8` and `swig==4.5.1`, then installed:

```text
gymnasium==1.4.0
pygame-ce==2.5.8
cloudpickle==3.1.2
farama-notifications==0.0.6
```

At this point the package installation portion of the Box2D workaround was
complete. Environment creation and stepping are verified separately because a
successful package import alone does not prove that a Gymnasium environment
can initialize and execute its physics.

## Meaning and tradeoff of `--no-build-isolation`

Normally, build isolation improves reproducibility by compiling a package with
only its declared build dependencies. Disabling isolation was appropriate here
because the source build needed the already installed Python-based SWIG tool.

The tradeoff is that the resulting build depends on the current environment's
tool versions. For this reason, the exact setuptools, wheel, packaging, SWIG,
Python, and Box2D versions are recorded above and in `versions.txt`.

No `sudo`, system-wide package installation, or modification of the operating
system Python was needed. All Python-facing tools remain contained in the
shared reinforcement-learning virtual environment.
