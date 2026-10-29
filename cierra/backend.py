"""Where the reference engine's arrays live: NumPy on the CPU (the default and the source of truth) or CuPy on an
NVIDIA GPU. The engine is written once; each function takes its array module from its input (`xp(code)`), so the
same float64 maths runs on either, and a GPU run is compared with the CPU run in the tests.

CuPy (MIT) is optional: `pip install -e ".[cuda]"` (it brings NVIDIA's CUDA runtime libraries from PyPI, under
NVIDIA's licence; nothing of it ships with Cierra). OpenColorIO has no CuPy path, so input and output transforms
(camera log, ACES 2.0 display) run on the CPU and the picture crosses to the GPU and back once per render.

    from cierra.backend import gpu_unavailable
    render(prog, image, device="gpu")      # or CIERRA_DEVICE=gpu for every render that does not say
"""

import os

import numpy as np

_CUPY = {}


def _cupy():
    """The cupy module, or None when it is not installed or finds no GPU (checked once)."""
    if "m" not in _CUPY:
        try:
            import cupy
            cupy.cuda.runtime.getDeviceCount()
            _CUPY["m"], _CUPY["why"] = cupy, None
        except Exception as e:  # ImportError, or a CUDA driver / runtime error
            _CUPY["m"], _CUPY["why"] = None, f"{type(e).__name__}: {e}"
    return _CUPY["m"]


def gpu_unavailable():
    """None when renders can run on the GPU here, else why not."""
    return None if _cupy() is not None else f"no CuPy GPU ({_CUPY['why']}); pip install -e \".[cuda]\""


def device(requested=None):
    """'cpu' or 'gpu': the requested device, else CIERRA_DEVICE, else 'cpu'. 'gpu' without a GPU is an error."""
    d = (requested or os.environ.get("CIERRA_DEVICE") or "cpu").lower()
    if d not in ("cpu", "gpu"):
        raise ValueError(f"device must be cpu or gpu, not {d!r}")
    if d == "gpu" and gpu_unavailable():
        raise RuntimeError(gpu_unavailable())
    return d


def is_gpu(a):
    cp = _CUPY.get("m")
    return cp is not None and isinstance(a, cp.ndarray)


def xp(*arrays):
    """The array module of the inputs: cupy when any is a GPU array, else numpy."""
    return _CUPY["m"] if any(is_gpu(a) for a in arrays) else np


def ndimage(*arrays):
    """scipy.ndimage, or cupyx.scipy.ndimage for GPU arrays (same filters, same edge modes)."""
    if any(is_gpu(a) for a in arrays):
        import cupyx.scipy.ndimage as nd
        return nd
    import scipy.ndimage as nd
    return nd


def const(v, like, dtype=np.float64):
    """A constant (scalar list, matrix) as an array on the same device as `like`."""
    return xp(like).asarray(v, dtype=dtype)


def to_host(a):
    """A NumPy array (copied from the GPU when needed)."""
    return _CUPY["m"].asnumpy(a) if is_gpu(a) else np.asarray(a)


def to_device(a, dev):
    """`a` as an array on 'cpu' or 'gpu'."""
    if dev == "gpu":
        return _cupy().asarray(a, dtype=np.float64)
    return to_host(a)
