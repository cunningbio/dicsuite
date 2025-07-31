import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import warnings

def load_shear_log(log_path, img_path):
    """
    Check shear angle logs for a matching record for the (singular) input image.

    Args:
        log_path (str or Path): Full path to the shear log CSV file.
        img_path (str): Full path to the image to be processed.

    Returns:
        pandas.DataFrame, float or None: The shear registry and logged shear angle for the image if found.
            May return empty data frame and None if log doesn't exist or if shear angle record not found.
    """
    # Ensure types are as needed!
    log_path = Path(log_path) / "shear_angle_log.csv"
    img_path = str(img_path)

    # Check if log exists - if so, read in!
    if log_path.exists():
        tqdm.write(f"Shear angle registry found at {str(log_path)}!")
        shear_df = pd.read_csv(log_path, index_col=0)

        # Ensure logs contain necessary columns and store matching registry
        if "Image_Path" not in shear_df.columns or "Shear_Angle" not in shear_df.columns:
            raise ValueError("Log file must contain 'Image_Path' and 'Shear_Angle' columns.")
        matching = shear_df[shear_df["Image_Path"] == img_path]

        # Next, check the match to ensure we have either 1 or 0 registers - if more than one, return an error
        if len(matching) == 1:
            tqdm.write("Shear angle found in registry! Skipping shear computation...")
            return shear_df, float(matching["Shear_Angle"].iloc[0])
        elif len(matching) > 1:
            raise ValueError("Multiple matching shear angles in registry — wipe and try again!")
        else:
            # If no register is found, return none
            tqdm.write("Shear angle not found in registry, computing...")
            return shear_df, None

    # If log doesn't exist, create a blank file to fill in once estimation is completed
    else:
        tqdm.write("No shear angle registry detected. Creating logs, then computing shear angle...")
        # Create empty directory
        log_path.parent.mkdir(parents=True, exist_ok=True)
        empty_log = pd.DataFrame(columns=["Image_Path", "Shear_Angle"])
        empty_log.to_csv(log_path)
        return empty_log, None


def update_shear_log(log_df, image_path, angle):
    tqdm.write(f"Shear angle rounded to: {str(round(angle / 45) * 45)}")
    shear_out = pd.DataFrame([{"Image_Path": image_path, "Shear_Angle": angle}])
    # If registry is empty, just capture the new output as the full registry
    shear_out = (shear_out.copy() if log_df.empty else pd.concat([log_df, shear_out], ignore_index=True))
    return shear_out

def save_shear_log(log_df, path):
    log_df.to_csv(path / "shear_angle_log.csv")

def create_collage_grid(images, parameters, out_dir, file_name, cmap='gray'):
    """
    Write out collage of images, based on combinations of reconstruction input parameters.

    Args:
        images (list): List of numpy arrays, assumed to be 8-bit grayscale.
        parameters (list): List of smoothing parameters.
        out_dir (str or Path): Full path to the output directory.
        cmap (str): String denoting colour map to use for Matplotlib plotting.

    Returns:
        None
    """
    # If only one parameter/image is passed, collage writing is not possible
    if len(images) == 1 or len(parameters) == 1:
        warnings.warn("Warning: Only one reconstructed image passed, no collage created.")
        return(None)

    import matplotlib.pyplot as plt
    # Extract numbers of rows/columns according to input parameters
    smooth_vals = sorted(set(s for s, _ in parameters))
    stabil_vals = sorted(set(st for _, st in parameters))
    n_rows = len(smooth_vals)
    n_cols = len(stabil_vals)

    # Instantiate empty subplot to build from - infer aspect ratio from first image to fit neatly
    ar = images[0].shape[0] / images[0].shape[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, (3 * n_rows) * ar), constrained_layout=True)

    # Now loop through input parameter combinations
    for image, (smooth, stabil) in zip(images, parameters):
        i = smooth_vals.index(smooth)
        j = stabil_vals.index(stabil)

        ax = axes[i, j]
        ax.imshow(image, cmap=cmap)
        ax.axis('off')
        #ax.set_title(f"Sm:{smooth}, St:{stabil}", fontsize=8)
        if i == 0:
            ax.set_title(f"Stable: {stabil}", fontsize=12)
        if j == 0:
            ax.text(-0.1, 0.5, f"Smooth: {smooth}", fontsize=12, va="center", ha="right", transform=ax.transAxes, rotation=90)
            #ax.set_ylabel(f"Smoothing: {smooth}", rotation=90, labelpad=10)

    #plt.tight_layout(rect=[0.1, 0, 1, 1])
    #plt.tight_layout()
    plt.savefig(out_dir / (str(file_name.stem) + "_collage.png"), dpi=300)
    plt.close()
