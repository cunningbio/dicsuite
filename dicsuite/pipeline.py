import cv2
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import numpy as np
from pathlib import Path

from dicsuite.metrics import analyse_shear
#from tqdm import tqdm
from tqdm.auto import tqdm
from typing import List, Optional
import warnings

from .backend import get_array_module, get_filter_backend
from .core import qpi_reconstruct, compute_shear, draw_shear_vector, create_psf, pad_fft
from .logging import load_shear_log, update_shear_log, save_shear_log, save_quality
from .metrics import analyse_contrast, analyse_sharpness, analyse_resolution
from .types import ImageData, GeneralConfig, ReconstructionConfig, SegmentationConfig
from .utils import param_to_str, ensure_list, ensure_numpy, prepare_output_folders, broadcast_param

SUPPORTED_EXTS = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}

## Supplementary functions used for image handling and processing
def _get_ImageData(input_path, recursive):
    """
    Given a list of input paths (files or dirs), returns an ImageData object.
    This includes image_path and source_path attributes for use later on.
    """
    # First, ensure that appropriate variables are iterable
    input_path = ensure_list(input_path)
    # Now loop through input files
    image_data_out = []
    for p in input_path:
        p = Path(p)
        if p.is_dir():
            globber = "**/*" if recursive else "*"
            for file in p.glob(globber):
                if file.suffix.lower() in SUPPORTED_EXTS:
                    image_data_out.append(ImageData(file.resolve(), p, _read_image))
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            image_data_out.append(ImageData(p.resolve(), p, _read_image))
        else:
            raise ValueError(f"Invalid input path: {p}")
    return image_data_out

def _get_output_dir(user_output=None):
    """Set output directory, depending on user preference."""
    if user_output:
        return Path(user_output)
    # Default: create 'dic_qpi_outputs/' next to current working directory
    base_dir = Path().resolve()
    return base_dir / "dic_qpi_outputs"

def _get_output_path(out_dir, smooth_i, stabil_i, smooth_len, stabil_len):
    """Set output file name, creating output subdirectories based on input parameters if needed."""
    # If smoothing parameter input is an array of to test (implied optimisation), need to create a subdirectory to write out to
    if smooth_len > 1:
        out_dir = (out_dir.with_suffix("") / param_to_str("Smooth", str(round(smooth_i, 8)))).with_suffix(".tiff")
    if stabil_len > 1:
        out_dir = (out_dir.with_suffix("") / param_to_str("Stab", str(round(stabil_i, 8)))).with_suffix(".tiff")
    Path(out_dir).parent.mkdir(parents=True, exist_ok=True)

    return out_dir

def _read_image(file):
    return cv2.imread(file, cv2.IMREAD_GRAYSCALE)

def _load_image_stack(file_list, batch_size=64, num_threads=8, recursive = False):
    """
    Load in either a single or list of image file(s) - multithread reading to reduce IO overhead.
    Outputs a list of 2D NumPy arrays.
    """
    # Firstly, handle user input to instantiate ImageData objects
    image_list = _get_ImageData(file_list, recursive)
    # Then instantiate blank list to append to as we work through batches

    ## Finally, load in the batches and read with looping
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Initialise chunks to loop through
        for i in tqdm(range(0, len(image_list), batch_size), desc="Loading image chunks"):
            batch_files = image_list[i:i+batch_size]
            # Now simply read through and append to the output list! No error catching here, but we can be permissive with input
            executor.map(lambda img: img.read(), batch_files)

    return image_list

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

# Main functions/subfunctions
def to_uint8(image):
    image = np.clip(image, 0, None)
    return (255 * image / image.max()).astype(np.uint8)

def contrast_adjust(images, xp, lower_q=0.25, upper_q=0.99):
    """
    Perform rudimentary contrast adjustment, based on quartile normalisation.
    Assumed 8-bit input.

    Args:
        images (list of ndarray): List of 2D (grayscale) images.
        lower_q (float): Lower quantile for minimum adjustment.
        upper_q (float): Upper quantile for maximum adjustment.

    Returns:
        list of ndarray: List of contrast-adjusted uint8 images.
    """
    single_image = isinstance(images, xp.ndarray)
    images = [images] if single_image else images
    # Flatten and concatenate all pixel values into one array
    flat_images = xp.concatenate([img.ravel() for img in images])

    # Compute global quantiles
    lower = xp.quantile(flat_images, lower_q)
    upper = xp.quantile(flat_images, upper_q)

    if upper == lower:
        raise ValueError("Global quantiles are equal; contrast adjustment would divide by zero.")

    # Apply the same transformation to each image
    adjusted_images = []
    for img in images:
        img_out = (img - lower)
        img_out[img_out < 0] = 0
        img_out = img_out * (255.0 / (upper - lower))
        img_out[img_out > 255] = 255
        adjusted_images.append(img_out)

    return adjusted_images[0] if single_image else adjusted_images


def qpi_reconstruct_batch(
        files_in: List[Path],
        out_dir: Optional[Path] = None,
        general_config: Optional[GeneralConfig] = None,
        reconstruction_config: Optional[ReconstructionConfig] = None,
        segmentation_config: Optional[SegmentationConfig] = None
):
    """
    Handles the batch or single running of the reconstruction algorithm.

    If infer_from_first is True, the shear angle and subsequent PSF are calculated from the first image and applied
        to all subsequent images. Input image dimensions will also be locked to first-in-sequence.

    Args:
        files_in (str, Path, or list/tuple of Path): One or more input image paths. If path is a directory, all files with image suffix are read.
        out_dir (str or Path, optional): Path to output directory. If not provided, one will be created in the parent directory of input image(s).
        general_config (GeneralConfig): General processing parameters.
            - run_cellpose (bool): If true, segmentation is carried out using parameters specified in segmentation_config.
            - recursive (bool): If true, directory input is searched through recursively to include all images.
            - infer_from_first (bool): If true, shear angle is to be computed only from first image, preferred for time-lapse image sequences.
            - contrast_adj (bool): If true, contrast of reconstructed images is adjusted to reduce noise
            - use_gpu (bool): True if GPU processing is preferred (default is False).
            - write_collage (bool): True if writing out a collage of reconstructed images - intended to QC input paramaters

        reconstruction_config (ReconstructionConfig): Reconstruction-specific parameters.
            - method (str): reconstruction method to implement
            - smooth_in (int or ndarray): Smoothing parameter. If multiple are passed, generate output for each specified.
            - stabil_in (int or ndarray): Stability parameter. If multiple are passed, generate output for each specified.
            - shear_angle (float or list, optional): Shear angle of DIC image in degrees. If not specified, shear angle is estimated.
            If working in stack mode, a single shear angle, if known, is expected, while for batch, a shear angle can be included for each image to reconstruct

        segmentation_config (SegmentationConfig): Segmentation-specific parameters - not implemented yet

    Returns:
        xp.ndarray: Reconstructed image.
            ndarray of type matching xp (e.g., numpy.ndarray or cupy.ndarray)
    """
    # First things first, set input as default if needed
    general_config = general_config or GeneralConfig()
    reconstruction_config = reconstruction_config or ReconstructionConfig()
    segmentation_config = segmentation_config or SegmentationConfig()

    # Determine backend to use and cast input parameters as iterable if needed
    xp = get_array_module(prefer_gpu=general_config.use_gpu)
    smooth_in = ensure_list(reconstruction_config.smooth_in)
    stabil_in = ensure_list(reconstruction_config.stabil_in)
    param_combinations = list(product(smooth_in, stabil_in)) # Create iterable set of input combinations for later
    # Read in image stack, returning populated ImageData objects
    images_in = _load_image_stack(files_in, recursive = general_config.recursive)

    ## Before further processing, we need to set location for writing and reading - needed for logs
    # Get path for output, depending on user input
    out_dir = _get_output_dir(out_dir)
    # Now create output directory/subdirectories if needed and store paths to dict
    out_dir = prepare_output_folders(out_dir)

    ## Instantiate the loop to process image stacks/batches
    ## Before looping...
    # Instantiate empty row for QC output
    qual_mets = []
    # Ensure that shear angle and file name input fits with number of images, and ensure iterable
    shear_angle = broadcast_param(reconstruction_config.shear_angle, len(images_in))
    iterable = zip(images_in, shear_angle)

    # Then, begin looping!
    for i, values in enumerate(tqdm(iterable, desc="Looping through images...", position=0)):
    #for i, values in enumerate(iterable):
        # Unpack the zipped variables
        img_data, shear = values
        # Before processing anything, normalise input image to scale between 0 and 1, and transfer to GPU as needed
        img = _preprocess_image(img_data._image)
        img = xp.asarray(img)  # CuPy or NumPy, depending on backend

        ## Shear angle handling
        # If shear angle is not provided, compute shear angles where needed
        if shear is None:
            # If not inferring from the first image, or if it's the first loop, get shear angle and image dimensions
            if not general_config.infer_from_first or i == 0:
                # Check for logs firstly
                shear_df, shear = load_shear_log(out_dir["logs"], img_data.image_path)
                # If no log exists, compute manually
                if shear is None:
                    shear = compute_shear(img)

                    # This is the perfect time to write out to registry and save an overlay image for shear angle QC
                    shear_df = update_shear_log(shear_df, img_data.image_path, shear)
                    save_shear_log(shear_df, out_dir["logs"])
                    # To create and write out the shear angle overlay image
                    shear_overlay = draw_shear_vector(img, shear)
                    shear_overlay = to_uint8(shear_overlay)
                    # Write out to the QC folder, using the input file name as a prefix - need to convert CuPy arrays to NumPy
                    cv2.imwrite((out_dir["qc"] / Path(img_data.image_path).stem).with_suffix(".tiff"),
                                ensure_numpy(shear_overlay))

                # Store first computed shear if to be used for stack processing
                if i == 0:
                    first_shear = shear

            # If inferring from first image and not first run, use the stashed shear angle for reconstruction
            else:
                shear = first_shear

        # If in stack mode, perform check to ensure stack images are all the same dimension and create PSF and smoothing kernels
        # This workflow will prevent the need for unnecessary reprocessing through the image stack
        if general_config.mode == "stack" and i == 0:
            # For the first image, store expected dimensions and create stock input fields
            ndim = img.shape
            # If running in stack, I'll also want to write out PSF and smoothing matrices to prevent computation each iteration

        # For images in a stack following the first, ensure dimensions conform
        elif general_config.mode == "stack":
            if img.shape != ndim:
                raise ValueError(f"Dimensions of image stack don't match! Check images 0 and {i}...")

        ## Now that critical variables are established, run the reconstruction itself
        # Store these variables as empty to begin with, to capture from first run if in stack
        psf_stack = None
        smooth_stack = None
        img_stack = []
        # To handle multiple input parameters, iterate through all combinations, checking parameter settings to infer write location
        for smooth_it, stabil_it in tqdm(param_combinations, desc="Reconstructing", total=len(param_combinations), position=1, leave=False):
            img_recon, psf_stack, smooth_stack = qpi_reconstruct(img, smooth_in=smooth_it, stabil_in=stabil_it, shear_angle=shear, psf_trans=psf_stack, smooth_trans=smooth_stack)

            # If contrast adjustment is desired and running in batch mode, adjust contrast according to single image pixel intensities
            if general_config.contrast_adj and general_config.mode == "batch":
                img_recon = contrast_adjust(img_recon, xp)

            # Compute quality metrics and write to dictionary for dataframe conversion in logging
            filter_module = get_filter_backend(xp)
            qual_mets.append({'File': str(img_data.image_path),
                              'Smoothing': smooth_it,
                              'Stability': stabil_it,
                              'Shear': shear,
                              'Shear Meta-accuracy': analyse_shear(img_recon, shear, xp, filter_module),
                              'Sharp': analyse_sharpness(img_recon, filter_module),
                              'Contrast': analyse_contrast(img_recon, filter_module),
                              'Resolution': analyse_resolution(img_recon, xp, filter_module)})

            # Before writing out, convert to 8-bit image format
            img_recon = to_uint8(ensure_numpy(img_recon))

            # Look for correct output directory before writing out reconstruction
            write_file = _get_output_path(img_data.get_output_path(out_dir["recon"]), smooth_it, stabil_it, len(smooth_in), len(stabil_in))
            # Write out to the QC folder, using the input file name as a prefix - need to convert CuPy arrays to NumPy
            cv2.imwrite(write_file, img_recon)
            # Finally, cache the reconstructed image to write out collage if needed
            img_stack.append(img_recon)

        if not general_config.mode == "stack":
            psf_matrix = None
            smooth_matrix = None

        # For each image, write out collage if specified
        if general_config.write_collage:
            from .logging import create_collage_grid
            create_collage_grid(img_stack, param_combinations, out_dir["qc"], img_data.image_path)

    # If contrast adjustment is desired and running in stack mode, adjust contrast according to global pixel intensities
    if general_config.contrast_adj and general_config.mode == "stack":
        img_recon = contrast_adjust(img_recon)

    # Once loop is completed, write out QC metrics
    save_quality(qual_mets, out_dir["qc"])
