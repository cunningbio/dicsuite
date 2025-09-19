from itertools import product
import math

from dicsuite.backend import get_array_module, get_filter_backend
from dicsuite.core import create_psf, pad_fft
from dicsuite.utils import ensure_list
from .base import BaseReconstruction

class YinFiltering(BaseReconstruction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def default_params(cls):
        return {"smoothing": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
                "stability": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
                "psf": None, "psf_trans": None, "smooth_trans": None}

    # Functions for handling iterating over multiple input parameters
    ## Create a sensible iterable from parameters, returning a list that matches parameter order in the iterable
    def param_combinations(self):
        smooth_in = ensure_list(self.params["smoothing"])
        stabil_in = ensure_list(self.params["stability"])
        return list(product(smooth_in, stabil_in))
    ## Return the number of parameter combinations for iterating
    def param_count(self):
        return len(ensure_list(self.params["smoothing"])) * len(ensure_list(self.params["stability"]))
    ## Catch the parameters from param_combinations while iterating
    def set_params(self, params):
        self.params["smoothing"], self.params["stability"] = params
    ## Return the current parameters while iterating, for QC
    def get_params(self):
        return {"Smoothing": self.params["smoothing"], "Stability": self.params["stability"]}

    # The actual reconstruction logic
    def run(self, image, shear_angle):
        """
        Reconstructs estimate of QPI from DIC image input using the ZZY algorithm.

        Args:
            image (ndarray): 2D grayscale image, preferably float32.
            shear_angle (float, optional): Shear angle of DIC image in degrees. If not specified, shear angle is estimated.

        Returns:
            xp.ndarray: Reconstructed image.
                ndarray of type matching xp (e.g., numpy.ndarray or cupy.ndarray)

        """
        # Determine whether or not to use CPU- or GPU-based modules
        xp = get_array_module(image)
        filter_module = get_filter_backend(xp)

        # Error catching and argument cleaning
        ## Cast shear angle to float if needed
        try:
            shear_angle = float(shear_angle)
        except (ValueError, TypeError):
            raise ValueError("shear_angle must be a numeric type or None")

        ## Convert image to 32-bit if needed
        if not xp.issubdtype(image.dtype, xp.floating):
            image = image.astype(xp.float32)

        # Begin with simple image adjustments
        ## For correcting for shear angle rounding errors, rotate the image to match the shear angle and set angle to 0°
        image_dim = image.shape
        image = filter_module.rotate(image, -shear_angle, reshape=True, mode="nearest")
        rerotate = shear_angle
        shear_angle = 0

        # If PSF or smoothing kernels aren't provided, create here and pad with 0s to match dimensions of input image
        # Padding ensures the kernel is centered and ready for Fourier transform
        if self.params["psf_trans"] is None:
            # Instantiate PSF matrix if needed and rotate along shear direction - with shear correction, this will be 0°, so no change
            psf = create_psf(self.params["psf"], shear_angle) # This function remains compatible with non-0° rotations if needed
            # As PSF rotation is locked to NumPy array, transfer to GPU if necessary
            psf = xp.array(psf)
            self.params["psf_trans"] = pad_fft(psf, image, xp)

        if self.params["smooth_trans"] is None:
            # Set smoothing kernel
            smooth = xp.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=float) / 8
            self.params["smooth_trans"] = pad_fft(smooth, image, xp)

        # Now transform the image itself!
        image_trans = image - xp.mean(image)
        image_trans = xp.fft.fftshift(xp.fft.fft2(image_trans, [(2 * image_trans.shape[0]) - 1, (2 * image_trans.shape[1]) - 1]))

        ## Implementing the reconstruction
        numerator = -(self.params["psf_trans"] * image_trans)
        denominator = (self.params["smoothing"] * self.params["smooth_trans"] * self.params["smooth_trans"] + self.params["stability"] - self.params["psf_trans"] * self.params["psf_trans"])
        image_recon = numerator / denominator  # Reconstruction formula with regularization

        ## Reverse the Fourier shift, undoing the previous `fftshift`
        # This prepares the frequency-domain data to be transformed back to the spatial domain
        image_recon = xp.fft.ifftshift(image_recon)
        image_recon = xp.fft.ifft2(image_recon)

        # The `real` part is extracted and rescaled to get the final reconstructed image in the spatial domain.
        recon_out = (image_recon.real - image_recon.real.min()) / (image_recon.real.max() - image_recon.real.min())

        # Trim padding - after transformation, padding has flipped, so extract the trailing rows/columns
        recon_out = recon_out[xp.arange((recon_out.shape[0] - image.shape[0]), recon_out.shape[0]), :][:,
                    xp.arange((recon_out.shape[1] - image.shape[1]), recon_out.shape[1])]

        # Rerotate image back to starting angle
        recon_out = filter_module.rotate(recon_out, rerotate, reshape=True, mode="nearest")
        # Cut  the image down to omit edges
        recon_out = recon_out[math.ceil((recon_out.shape[0] - image_dim[0]) / 2):math.ceil(recon_out.shape[0] - ((recon_out.shape[0] - image_dim[0]) / 2)),
                    math.ceil((recon_out.shape[1] - image_dim[1]) / 2):math.ceil(recon_out.shape[1] - ((recon_out.shape[1] - image_dim[1]) / 2))]

        return recon_out

    def wipe(self):
        self.params["psf_trans"] = None
        self.params["smooth_trans"] = None
