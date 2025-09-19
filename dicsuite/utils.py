from pathlib import Path
import math

def param_to_str(name, value):
    """
    Helper function to format input parameters into filenames.
    """
    # Realistically, minus values aren't likely to be passed here - commented out handling for potential future use-cases!
    val = str(value).replace('.', '_')
    return f"{name}-{val}"

def grid_shape(n):
    """
    Helper function to format input parameters into filenames.
    """
    rows = int(math.floor(math.sqrt(n)))
    cols = math.ceil(n / rows)
    return rows, cols

# To cast input as iterable if needed
def ensure_list(x):
    """
    Cast input as iterable. Needed to run pipelines that can handle one and many input(s).

    Args:
        x (any): Any variable that will be iterated on.

    Returns:
        list or tuple: x cast as list, or x unchanged if x is a tuple.
    """
    return x if isinstance(x, (list, tuple)) else [x]

def ensure_numpy(array):
    """
    Facilitates the conversion of CuPy to NumPy arrays.
    Needed to write out images where GPU processing is preferred.
    """
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    except ImportError:
        pass
    return array


def broadcast_param(x, target_len):
    """
    Constrain list to length, or create repeating list of desired length for single items.

    Args:
        x (any or list/tuple): Single item variable or list/tuple.
        target_len (int): Desired length of output list.

    Returns:
        list or tuple: x as list repeated to input length, or x unchanged if x was originally of correct length.
    """
    x = ensure_list(x)
    if len(x) == 1:
        return x * target_len
    elif len(x) == target_len:
        return x
    else:
        raise ValueError("Length of input parameters does not match number of images!")


def prepare_output_folders(base_path):
    """
    Create base output directory and nested folders for logs, QC, and recon results.

    Args:
        base_path (Path or str): Path to main output directory (e.g. 'dic_qpi_outputs/')

    Returns:
        dict: A dictionary of paths to output directory/subdirectories.
    """
    base = Path(base_path)
    folders = ['logs', 'qc', 'recon']

    for sub in folders:
        path = base / sub
        path.mkdir(parents=True, exist_ok=True)

    return {
        "base": base,
        "logs": base / "logs",
        "qc": base / "qc",
        "recon": base / "recon"
    }


