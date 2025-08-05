import math
import numpy as np
from skimage.draw import line
from skimage.transform import rotate as sk_rotate

from .backend import get_array_module, get_filter_backend
from .utils import ensure_list

## All defined functions assume images are normalised with pixel intensities between 0-1
def linear_strel(se_length, se_angle, xp = None):
    """
    Creates a linear structural element with defined length/angle.
    Matches use case of MATLAB's strel function, where structural element is centred and output array dimensions are shaped according to defined length.

    Args:
        se_length (int): 2D grayscale image, preferably float32.
        se_angle (int): Angle of shear in degrees.
        xp (module, optional): Array-processing module to use (e.g., numpy or cupy).
            If None, NumPy is used by default.

    Returns:
        xp.ndarray: Linear structural element.
            ndarray of type matching xp (e.g., numpy.ndarray or cupy.ndarray)
    """
    # Determine CPU/GPU processing
    if xp is None:
        xp = np

    # If the angle supplied is > 180, squash to fit between 0-180
    se_angle = se_angle if (se_angle <= 180) else se_angle - (180 * (se_angle // 180))
    # Calculate the start and end points of the line to fit the array
    angle_rad = np.deg2rad(se_angle) # Convert to radians
    x2 = abs(round(se_length * np.cos(angle_rad)))
    y2 = abs(round(se_length * np.sin(angle_rad)))
    # Ensure dimensions are odd to define a proper center
    x2 = x2 + 1 if x2 % 2 == 1 else x2
    y2 = y2 + 1 if y2 % 2 == 1 else y2

    # Create a zero array with calculated dimensions
    strel = xp.zeros((y2 + 1, x2 + 1), dtype=xp.uint8)

    # Generate line coordinates
    rr, cc = line(0, x2, y2, 0) if se_angle < 90 else line(0, 0, y2, x2)
    # Set the line in the zero array
    strel[rr, cc] = 1

    return strel

## Array handling functions, for PSF and smoothing kernels
def pad_fft(arr, img, xp):
    """
    Helper function to pad arrays for fast fourier transform.
    """
    arr_pad = xp.pad(arr, (
        (img.shape[0] - math.ceil(arr.shape[0] / 2), img.shape[0] - math.ceil(arr.shape[0] / 2)),
        (img.shape[1] - math.ceil(arr.shape[1] / 2), img.shape[1] - math.ceil(arr.shape[1] / 2))),
                     constant_values=0)
    return xp.fft.fftshift(xp.fft.fft2(arr_pad))

def create_psf(shear_angle):
    if shear_angle % 45 != 0:
        raise ValueError("Shear angle must be multiple of 45 degrees to create PSF!")
    psf = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=float) / 2
    psf = sk_rotate(psf, 180 + shear_angle, resize=False, order=0)  # Setting order to 0 gets around need to check 45 degree compatibility
    return(psf)


def compute_shear(image, von_mises = True):
    """
    Reconstructs estimate of QPI from DIC image input using the ZZY algorithm.
    Handles both CPU and GPU arrays depending on `use_gpu` flag.

    Args:
        image (ndarray): 2D grayscale image, assumed normalised and float32.
        use_gpu (bool): True if GPU processing is preferred (default is False).
        von_mises (bool): True if inferring optimal shear angle using von Mises distribution,
            otherwise return max value (default is True)

    Returns:
        int: Computed shear angle.

    """
    ## Adjustable parameters here
    n_dir = 180  # Number of directions
    l_kern = 20  # Length of kernel

    # Get module for handling numeric data, using input image type to infer CPU or GPU processing
    xp = get_array_module(image)
    filter_module = get_filter_backend(xp)

    # Initialize gradient sum storage
    gradient_sum = np.zeros(n_dir)
    for x in range(n_dir):
        angle_in = (x * 180) / n_dir
        # Create structuring element (kernel) to for dilation/erosion - note that lengths for diagonals may be off by sqrt(2)
        se = linear_strel(l_kern, angle_in, xp)
        dilated = filter_module.grey_dilation(image, footprint=se)
        eroded = filter_module.grey_erosion(image, footprint=se)
        gradient = dilated - eroded

        # Sum of all gradient pixels
        gradient_sum[x] = xp.sum(gradient)

    ## Estimate shear angle depending on approach specified
    if von_mises:
        # Fit von Mises distribution and extract mu
        from scipy.stats import vonmises
        # Shift data so minimum is at angle 0°
        min_grad = np.argmin(gradient_sum)
        grad_shifted = np.concatenate((gradient_sum[min_grad:], gradient_sum[:min_grad]))

        # Expand the shifted frequency distribution into data points
        circ_data = np.deg2rad(np.repeat(np.arange(0, 180), grad_shifted.astype(int)))  # Convert to radians too!

        # Fit the Von Mises distribution and extract mu
        circ_data = vonmises.fit(circ_data, fscale=1)[1]  # fix scale=1 for circular
        # Finally, convert to degree and re-adjust based on the original shift
        return((np.rad2deg(circ_data) + min_grad) % 180)  # Use modulus to wrap back to 0–180 domain
    else:
        # If not specified, just extract the maximum of the distribution
        return((np.argmax(gradient_sum) * 180) / n_dir)

def draw_shear_vector(image, shear_angle):
    """
    Overlays a vector at an angle corresponding to computed shear onto input image array.

    Args:
        image (numpy.ndarray or cupy.ndarray): Image array to overlay shear vector onto. Assumed to be normalised.
        shear_angle (int): Shear angle to overlay

    Returns:
        numpy.ndarray or cupy.ndarray: Grayscale image array with shear vector overlay.
            ndarray of type matching input array.
    """
    # Get module for handling numeric data, using input image type to infer CPU or GPU processing
    xp = get_array_module(image)
    # Overlay shear angle indicator by creating an appropriate line matrix and expand to match dim of input image
    shear_overlay = linear_strel(min(image.shape) - 2, shear_angle, xp)  # Use smaller of row/column sizes, so no possibility of linear matrix outsizing input
    shear_overlay = xp.pad(shear_overlay, ((math.ceil((image.shape[0] - shear_overlay.shape[0]) / 2),
                                            math.floor((image.shape[0] - shear_overlay.shape[0]) / 2)),
                                           (math.ceil((image.shape[1] - shear_overlay.shape[1]) / 2),
                                            math.floor((image.shape[1] - shear_overlay.shape[1]) / 2))), constant_values=0)

    ### Write out
    shear_to_out = image + shear_overlay  # Input images should have 0 values min and 1 values max! Simple addition works.
    shear_to_out = np.clip(shear_to_out, 0, 1)

    return(shear_to_out)

def qpi_reconstruct(image, smooth_in = 1, stabil_in = 0.0001, shear_angle=None, psf_trans = None, smooth_trans = None, rotate_correct = True):
    """
    Reconstructs estimate of QPI from DIC image input using the ZZY algorithm.
    Handles both CPU and GPU arrays depending on `use_gpu` flag.

    Args:
        image (ndarray): 2D grayscale image, preferably float32.
        smooth_in (int): Smoothing parameter.
        stabil_in (int): Stability parameter.
        shear_angle (float, optional): Shear angle of DIC image in degrees. If not specified, shear angle is estimated.
        psf_trans (ndarray, optional): PSF matrix, padded and Fourier transformed.
        smooth_trans (ndarray, optional): Smoothing kernel, padded and Fourier transformed.
        rotate_correct (bool): True if rotating input image to correct for shear angles that are not multiples of 45° (default is True).


    Returns:
        xp.ndarray: Reconstructed image.
            ndarray of type matching xp (e.g., numpy.ndarray or cupy.ndarray)

    """
    # Determine whether or not to use CPU- or GPU-based modules
    xp = get_array_module(image)
    filter_module = get_filter_backend(xp)

    ## Error catching and argument cleaning
    # Cast shear angle to float if needed
    if shear_angle is not None:
        try:
            shear_angle = float(shear_angle)
        except (ValueError, TypeError):
            raise ValueError("shear_angle must be a numeric type or None")
    else:
        shear_angle = compute_shear(image)

    # Convert image to 32-bit if needed
    if not xp.issubdtype(image.dtype, xp.floating):
        image = image.astype(xp.float32)

    ## Begin with simple image adjustments
    # If correcting for shear angle rounding errors, rotate the image to match the shear angle and set angle to 0
    if rotate_correct:
        image_dim = image.shape
        image = filter_module.rotate(image, -shear_angle, reshape=True, mode = "nearest")
        rerotate = shear_angle
        shear_angle = 0

    # Round the estimated direction to the nearest multiple of 45 degrees
    shear_angle = np.round(shear_angle / 45) * 45

    # If PSF or smoothing kernels aren't provided, create here and pad with 0s to match dimensions of input image
    # Padding ensures the kernel is centered and ready for Fourier transform
    if psf_trans is None:
        # Instantiate PSF matrix and rotate along shear direction
        psf = create_psf(shear_angle)
        # As PSF rotation is locked to NumPy array, transfer to GPU if necessary
        psf = xp.array(psf)
        psf_trans = pad_fft(psf, image, xp)

    if smooth_trans is None:
        # Set smoothing kernel
        smooth = xp.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=float) / 8
        smooth_trans = pad_fft(smooth, image, xp)

    ## Now transform the image itself!
    image_trans = image - xp.mean(image)
    image_trans = xp.fft.fftshift(xp.fft.fft2(image_trans, [(2 * image_trans.shape[0]) - 1, (2 * image_trans.shape[1]) - 1]))
    image_recon = -(psf_trans * image_trans) / (smooth_in * smooth_trans * smooth_trans + stabil_in - psf_trans * psf_trans)  # Reconstruction formula with regularization

    ## Reverse the Fourier shift, undoing the previous `fftshift`
    # This prepares the frequency-domain data to be transformed back to the spatial domain
    image_recon = xp.fft.ifftshift(image_recon)
    image_recon = xp.fft.ifft2(image_recon)

    # The `real` part is extracted and rescaled to get the final reconstructed image in the spatial domain.
    recon_out = (image_recon.real - image_recon.real.min()) / (image_recon.real.max() - image_recon.real.min())

    # Trim padding - after transformation, padding has flipped, so extract the trailing rows/columns
    recon_out = recon_out[xp.arange((recon_out.shape[0] - image.shape[0]), recon_out.shape[0]), :][:,
                xp.arange((recon_out.shape[1] - image.shape[1]), recon_out.shape[1])]

    # If rotated, rerotate back into the starting angle
    if rotate_correct:
        recon_out = filter_module.rotate(recon_out, rerotate, reshape=False)
        # Cut  the image down to omit edges
        recon_out = recon_out[math.ceil((recon_out.shape[0] - image_dim[0]) / 2):math.ceil(
            recon_out.shape[0] - ((recon_out.shape[0] - image_dim[0]) / 2)),
                    math.ceil((recon_out.shape[1] - image_dim[1]) / 2):math.ceil(
                        recon_out.shape[1] - ((recon_out.shape[1] - image_dim[1]) / 2))]
        shear_angle = rerotate  # Output complete, non-rounded shear angle for future work!

    return recon_out, psf_trans, smooth_trans