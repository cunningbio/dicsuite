from .base import BaseReconstruction
from dicsuite.core import get_gauss_deriv, wiener_filter
from dicsuite.utils import ensure_list,ensure_numpy
from dicsuite.backend import get_array_module

class WienerFiltering(BaseReconstruction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def default_params(cls):
        return {"kernel_sigma": [0.1, 0.25, 0.5, 0.75],
                "nsr": None, "psf": None}

    # Functions for handling iterating over multiple input parameters
    ## Create a sensible iterable from parameters
    def param_combinations(self):
        return ensure_list(self.params["kernel_sigma"])
    ## Return the number of parameter combinations for iterating
    def param_count(self):
        return len(ensure_list(self.params["kernel_sigma"]))
    ## Catch the parameters from param_combinations while iterating
    def set_params(self, params):
        self.params["kernel_sigma"] = params
    ## Return the current parameters while iterating, for QC
    def get_params(self):
        return {"KernelSigma": self.params["kernel_sigma"]}

    # Run the reconstruction
    def run(self, image, shear_angle):
        # Determine whether or not to use CPU- or GPU-based modules
        xp = get_array_module(image)
        ## Convert image to 32-bit if needed
        if not xp.issubdtype(image.dtype, xp.floating):
            image = image.astype(xp.float32)

        # Set PSF from input
        if self.params["psf"] is None:
            psf = get_gauss_deriv(sigma=self.params["kernel_sigma"], direction=shear_angle, xp = xp)
            psf = psf / xp.sum(xp.abs(psf))
            psf[xp.logical_and(psf >= 0,
                               psf < 1e-9)] = 1e-9  # Adjusts tiny values to minimum accepted by Wiener filter - less representative of truth...
            psf[xp.logical_and(psf <= 0,
                               psf > -1e-9)] = -1e-9  # Adjusts tiny values to minimum accepted by Wiener filter - less representative of truth...
            self.params["psf"] = psf
        if self.params["nsr"] is None:
            self.params["nsr"] = 0.01

        img_out = wiener_filter(-image, self.params["psf"], self.params["nsr"], xp)  # Wiener filter breaks if zeros (or numbs < 1e-10) are included in PSF...
        return img_out

    # Wipe relevant parameters for non-stack runs
    def wipe(self):
        self.params["psf"] = None
        self.params["nsr"] = None
