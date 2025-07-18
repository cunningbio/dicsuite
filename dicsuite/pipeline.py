import cv2
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
import warnings

from .backend import get_array_module
from .core import qpi_reconstruct, compute_shear, draw_shear_vector, create_psf, pad_fft
from .utils import param_to_str, ensure_list, ensure_numpy, prepare_output_folders, broadcast_param
from .logging import load_shear_log, update_shear_log, save_shear_log

## Supplementary functions used for image handling and processing
def _resolve_paths(input_path, ext_in = ("*.tif", "*.tiff", "*.png")):
    """Expands input path(s) into a list of image file paths."""
    # First, ensure that appropriate variables are iterable
    input_path = ensure_list(input_path)
    ext_in = ensure_list(ext_in)
    # Now loop through input files
    paths = []
    for p in input_path:
        p = Path(p)
        if p.is_dir():
            for pattern in ext_in:
                paths.extend(sorted(p.glob(pattern)))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"Path not found: {p}")
    return paths

def _get_output_dir(image_paths, user_output=None):
    """Set output directory, depending on user preference."""
    if user_output:
        return Path(user_output)
    # Default: create 'dic_qpi_outputs/' next to input image folder
    base_dir = Path(image_paths[0]).parent
    return base_dir / "dic_qpi_outputs"

def _get_output_path(file_name, out_dir, smooth_i, stabil_i, smooth_len, stabil_len):
    """Set output file name, creating output subdirectories based on input paramaters if needed."""
    # If smoothing parameter input is an array of to test (implied optimisation), need to create a subdirectory to write out to
    if smooth_len > 1:
        out_dir = out_dir / param_to_str("Smooth",str(round(smooth_i, 8)))
    elif stabil_len > 1:
        out_dir = out_dir / param_to_str("Stab",str(round(stabil_i, 8)))
    else:
        out_dir = out_dir / file_name

    write_dir = out_dir.with_suffix(".tiff")
    Path(write_dir).mkdir(parents=True, exist_ok=True)

    return write_dir

def _read_image(file):
    return cv2.imread(file, cv2.IMREAD_GRAYSCALE)

def _load_image_stack(file_list, batch_size=64, num_threads=8):
    """
    Load in either a single or list of image file(s) - multithread reading to reduce IO overhead.
    Outputs a list of 2D NumPy arrays.
    """
    # Firstly, handle user input and read in images
    file_list = _resolve_paths(file_list)
    # Then instantiate blank list to append to as we work through batches
    image_out = []

    ## Finally, load in the batches and read with looping
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Initialise chunks to loop through
        for i in tqdm(range(0, len(file_list), batch_size), desc="Loading image chunks"):
            batch_files = file_list[i:i+batch_size]
            # Now simply read through and append to the output list! No error catching here, but we can be permissive with input
            batch = list(executor.map(_read_image, batch_files))
            image_out.append(batch)

    return image_out

import numpy as np

def _preprocess_image(image):
    """
    Preprocesses input images, normalising pixel values from 0-1
    """
    # Check if image is 2D array - only 3D images supported so far are polymerized structure dataset
    if image.ndim != 2:
        if image.ndim == 3:
            warnings.warn("Warning: 3D image included as input, RGB reconstruction not currently optimised! Slicing to first colour and handling as 2D grayscale.")
            image = image[:, :, 0]
        else:
            raise ValueError("Input image must be 2D (grayscale)")

    # Convert to float32 before normalization
    image = image.astype(np.float32)

    # Min-max scale to 0–1
    img_min, img_max = image.min(), image.max()
    if img_max != img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        image[:] = 0.0  # flat image

    return image

def to_uint8(image):
    image = np.clip(image, 0, None)
    return (255 * image / image.max()).astype(np.uint8)

def qpi_reconstruct_batch(files_in, out_dir = None,smooth_in = 1, stabil_in = 0.0001, mode = "stack", infer_from_first = True, shear_angle = None, use_gpu = False):
    """
    Handles the batch or single running of the reconstruction algorithm.

    If infer_from_first is True, the shear angle and subsequent PSF are calculated from the first image and applied
    to all subsequent images. Input image dimensions will also be locked to first-in-sequence.

    Args:
        files_in (str, Path, or list/tuple of Path): One or more input image paths.
        out_dir (str or Path, optional): Path to output directory. If not provided, one will be created in the parent directory of input image(s).
        smooth_in (int or ndarray): Smoothing parameter. If multiple are passed, generate output for each specified.
        stabil_in (int or ndarray): Stability parameter. If multiple are passed, generate output for each specified.
        mode (str): Angle of shear in degrees.
        infer_from_first (bool): True if shear angle, preferred for time-lapse image sequences.
        shear_angle (float or list, optional): Shear angle of DIC image in degrees. If not specified, shear angle is estimated.
            If working in stack mode, a single shear angle, if known, is expected, while for batch, a shear angle can be included for each image to reconstruct
        rotate_correct (bool): True if rotating input image to correct for shear angles that are not multiples of 45° (default is True).
        use_gpu (bool): True if GPU processing is preferred (default is False).

    Returns:
        xp.ndarray: Reconstructed image.
            ndarray of type matching xp (e.g., numpy.ndarray or cupy.ndarray)

    """
    # First things first, determine backend to use and cast input parameters as iterable if needed
    xp = get_array_module(prefer_gpu=use_gpu)
    smooth_in = ensure_list(smooth_in)
    stabil_in = ensure_list(stabil_in)
    # Read in image stack
    images_in = _load_image_stack(files_in)

    ## Before further processing, we need to set location for writing and reading - needed for logs
    # Get path for output, depending on user input
    out_dir = _get_output_dir(files_in, out_dir)
    # Now create output directory/subdirectories if needed and store paths to dict
    out_dir = prepare_output_folders(out_dir)

    ## Instantiate the loop to process image stacks/batches
    # Before looping, ensure that shear angle input fits with number of images, and ensure iterable
    shear_angle = broadcast_param(shear_angle, len(images_in))
    iterable = zip(images_in, ensure_list(files_in), shear_angle)

    # Then, begin looping!
    for i, values in enumerate(iterable):
        # Unpack the zipped variables
        img, file_name, shear = values
        # Before processing anything, normalise input image to scale between 0 and 1, and transfer to GPU as needed
        img = _preprocess_image(img)
        image = xp.asarray(image)  # CuPy or NumPy, depending on backend

        ## Shear angle handling
        # If shear angle is not provided, compute shear angles where needed
        if shear is None:
            # If not inferring from the first image, or if it's the first loop, get shear angle and image dimensions
            if not infer_from_first or i == 0:
                shear_df, shear = load_shear_log(out_dir["logs"], file_name)
                if shear is None:
                    shear = compute_shear(img)
                # Store first computed shear if to be used for stack processing
                if i == 0:
                    first_shear = shear
                ## This is also the perfect time to write out to registry and save an overlay image for shear angle QC
                # Write to registry
                shear_df = update_shear_log(shear_df, file_name, shear)
                save_shear_log(out_dir["logs"], shear_df)
                # To create and write out the shear angle overlay image
                shear_overlay = draw_shear_vector(img, shear)
                shear_overlay = to_uint8(shear_overlay)
                # Write out to the QC folder, using the input file name as a prefix - need to convert CuPy arrays to NumPy
                cv2.imwrite((out_dir["qc"] / Path(file_name).stem).with_suffix(".tiff"), ensure_numpy(shear_overlay))
            # If
            else:
                shear = first_shear

        # If in stack mode, perform check to ensure stack images are all the same dimension and create PSF and smoothing kernels
        if mode == "stack" and i == 0:
            # For the first image, store expected dimensions and create stock input fields
            ndim = img.shape
            # Create a PSF to feed into all reconstructions
            psf_share = create_psf(shear)
            psf_share = pad_fft(psf_share,img,xp)
            # And do the same for a smoothing kernel
            smooth_share = pad_fft(xp.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=float) / 8, img, xp)
        # For images in a stack following the first, ensure dimensions conform
        elif mode == "stack":
            if img.shape != ndim:
                raise ValueError(f"Dimensions of image stack don't match! Check images 0 and {i}...")

        ## Now that critical variables are established, run the reconstruction itself
        # To handle multiple input parameters, run a nested loop, checking parameter settings to infer write location
        for smooth_it in smooth_in:
            for stabil_it in stabil_in:
                if mode == "stack":
                    img_recon = qpi_reconstruct(img, smooth_in=smooth_it, stabil_in=stabil_it, shear_angle=shear,
                                             psf_trans=psf_share, smooth_trans=smooth_share)
                else:
                    img_recon = qpi_reconstruct(img, shear_angle=shear)
                # Look for correct output directory before writing out reconstruction
                write_file = _get_output_path(file_name, out_dir["recon"], smooth_it, stabil_it, len(smooth_in), len(stabil_in))
                img_recon = to_uint8(img_recon)
                # Write out to the QC folder, using the input file name as a prefix - need to convert CuPy arrays to NumPy
                cv2.imwrite(write_file, ensure_numpy(img_recon))


