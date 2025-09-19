from .core import linear_strel

## Helper functions to optionally measure various quality metrics for reconstruction.
def analyse_sharpness(image_in, mod):
    lap_in = mod.laplace(image_in)
    return lap_in.var()

def analyse_contrast(image_in, mod):
    lap_in = mod.laplace(image_in)
    # Clarity (using a simple combination of sharpness and contrast)
    return lap_in.var() * image_in.std()

def analyse_resolution(image_in, xp, mod):
    # Resolution (using Sobel operator)
    sobel_x = mod.sobel(image_in, axis=1)
    sobel_y = mod.sobel(image_in, axis=0)
    sobel_out = xp.sqrt(sobel_x**2 + sobel_y**2)
    return xp.mean(sobel_out)

def analyse_shear(image_in, shear_angle, xp, mod):
    """
    Returns the shear resolution, or the difference in gradient calculated at vs perpindicular to the shear angle
    """
    # Create structuring element (kernel) to for dilation/erosion - note that lengths for diagonals may be off by sqrt(2)
    se_corr = linear_strel(20, shear_angle, xp)
    se_incorr = linear_strel(20, shear_angle + 90, xp)
    gradient_corr = mod.grey_dilation(image_in, footprint = se_corr) - mod.grey_erosion(image_in, footprint = se_corr)
    gradient_incorr = mod.grey_dilation(image_in, footprint = se_incorr) - mod.grey_erosion(image_in, footprint = se_incorr)

    # Return the difference between the two
    return xp.sum(gradient_corr) - xp.sum(gradient_incorr)
