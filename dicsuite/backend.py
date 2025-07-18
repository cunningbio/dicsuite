
def get_array_module(x=None, prefer_gpu=False):
    """
    Returns the appropriate array module (NumPy or CuPy) depending on preference or input type.
    If `x` is provided, its type is used to determine backend.
    """
    try:
        import cupy as cp
        has_cupy = True
    except ImportError:
        cp = None
        has_cupy = False

    import numpy as np

    if x is not None:
        if has_cupy and isinstance(x, cp.ndarray):
            return cp
        else:
            return np

    return cp if prefer_gpu and has_cupy else np

def get_filter_backend(xp=None):
    """
    Returns the appropriate image filtering module (SciPy with or without CuPy compatibility) depending on preference or input type.
    Output of get_array_module and package availability can be used to determine backend.
    """
    import scipy.ndimage as snd
    try:
        import cupyx.scipy.ndimage as cnd
    except ImportError:
        cnd = None

    if xp and xp.__name__.startswith("cupy") and cnd:
        return cnd
    return snd
