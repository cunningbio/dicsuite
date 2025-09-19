import math
import numpy as np
from skimage.draw import line
from skimage.transform import rotate as sk_rotate

from .backend import get_array_module, get_filter_backend

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

# Helper function for Gaussian-derivative PSF generating
def get_gauss_deriv(size=None, sigma=0.5, direction=0, xp = None):
    # If no backend is passed, default to NumPy processing
    if xp is None:
        xp = np
    direction = math.pi - xp.deg2rad(direction)  # Convert to radians
    if size is None:
        size = math.ceil(6 * sigma + 1)
        if size % 2 == 0:
            size = size + 2
    # Creating kernel
    gauss_out = xp.array([]).reshape(0, size)
    for i in xp.arange(size) + 1:
        row_out = i - math.ceil(size / 2)
        col_out = [j - math.ceil(size / 2) for j in xp.arange(size) + 1]
        to_out = xp.array(
            [-x * math.exp(-(x ** 2 + row_out ** 2) / sigma ** 2) * math.cos(direction) for x in col_out])
        to_out = to_out + xp.array(
            [-row_out * math.exp(-(x ** 2 + row_out ** 2) / sigma ** 2) * math.sin(direction) for x in col_out])
        gauss_out = xp.concatenate((gauss_out, to_out.reshape(1, size)), axis=0)
    gauss_out = gauss_out / xp.sum(xp.abs(gauss_out))
    gauss_out[
        xp.abs(gauss_out) < xp.finfo(xp.float64).eps] = 0
    return (gauss_out)


def wiener_filter(image, psf, balance, xp, value_range=(-1, 1)):
    """
    CuPy-compatible Wiener filter (single-channel).

    Args:
        image (ndarray): 2D grayscale image, assumed normalised and float32.
        use_gpu (bool): True if GPU processing is preferred (default is False).
        von_mises (bool): True if inferring optimal shear angle using von Mises distribution,
            otherwise return max value (default is True)

    Returns:
        int: Computed shear angle.

    Args:
        image (numpy.ndarray or cupy.ndarray): 2D grayscale image, assumed normalised and float32.
        psf (numpy.ndarray or cupy.ndarray): Point spread function (same size or broadcastable to image).
        balance (float): Regularization constant (maps to `balance` in skimage's wiener function).
        value_range (tuple): Min/max clamp values, to match skimage.

    Returns:
        numpy.ndarray or cupy.ndarray: Wiener-deconvolved image.

    """
    # Ensure arrays are CuPy
    image = xp.asarray(image, dtype=xp.float32)
    psf = xp.asarray(psf, dtype=xp.float32)

    # Pad PSF to match image size
    pad_shape = [(0, s - ps) for (s, ps) in zip(image.shape, psf.shape)]
    psf_padded = xp.pad(psf, pad_shape, mode='constant')
    psf_padded = xp.pad(psf, pad_shape, mode='constant')
    psf_padded = xp.roll(psf_padded, shift=(-psf.shape[0]//2, -psf.shape[1]//2), axis=(0,1))

    # FFTs
    image_fft = xp.fft.fft2(image)
    psf_fft = xp.fft.fft2(psf_padded)

    # Wiener filter in frequency domain
    psf_conj = xp.conj(psf_fft)
    denom = xp.abs(psf_fft)**2 + balance
    result_fft = psf_conj / denom * image_fft

    # Back to spatial domain
    result = xp.fft.ifft2(result_fft).real

    # Clip output
    result = xp.clip(result, value_range[0], value_range[1])
    return result

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

def create_psf(psf ,shear_angle):
    if shear_angle % 45 != 0:
        raise ValueError("Shear angle must be multiple of 45 degrees to create PSF!")
    if psf is None:
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
