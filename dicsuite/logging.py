import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import warnings

from .utils import grid_shape

# Static list of possible input parameters for QC logging
ALL_PARAMS = [
    "Smoothing",
    "Stability",
    "KernelSigma",
]

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

# Helper function for flattening parameter dictionaries
def flatten_qc_record(record):
    params = record.pop("Params", {})
    return {**record, **params}

def save_quality(qual_df, path):
    # Firstly, save as data frame from the list of dictionaries and create registry path from log path
    qual_df = pd.DataFrame([flatten_qc_record(df_row) for df_row in qual_df])
    ## Need to create parameter entries for methods other than current
    for col in ALL_PARAMS:
        if col not in qual_df.columns:
            qual_df[col] = pd.NA

    reg_path = path / "recon_metrics.csv"
    # Next, if a previous registry was saved, read this in to update
    if reg_path.exists():
        old_df = pd.read_csv(reg_path)
        qual_df = pd.concat([old_df, qual_df], ignore_index=True)

    # Drop duplicates based on file names and input parameters
    subset_cols = ["File", "Method"] + ALL_PARAMS
    qual_df = qual_df.drop_duplicates(subset=subset_cols, keep="last")
    # Sort and write to CSV
    qual_df = qual_df.sort_values(by=subset_cols)
    qual_df.to_csv(reg_path, index=False)



def create_collage_grid(images, parameters, out_dir, file_name, cmap='gray'):
    """
    Write out collage of images, based on combinations of reconstruction input parameters.

    Args:
        images (list): List of numpy arrays, assumed to be 8-bit grayscale.
        parameters (list): List of smoothing parameter dictionaries.
        out_dir (str or Path): Full path to the output directory.
        file_name (Path): Full path to input file, used to create name for output collage.
        cmap (str): String denoting colour map to use for Matplotlib plotting.

    Returns:
        None
    """
    # Import needed packages
    import matplotlib.pyplot as plt
    # If only one parameter/image is passed, collage writing is not possible
    if len(images) == 1 or len(parameters) == 1:
        warnings.warn("Warning: Only one reconstructed image passed, no collage created.")
        return(None)

    # Extract numbers of rows/columns according to input parameters
    if parameters and len(parameters[0]) == 2:
        # If 2 dimensional gridding is possible, set up precursors for executing
        row_vals = []
        col_vals = []
        for params in parameters:
            values = list(params.values())  # preserves the original insertion order
            row_vals.append(values[0])
            col_vals.append(values[1])
        row_vals = sorted(set(row_vals))
        col_vals = sorted(set(col_vals))
        n_rows = len(row_vals)
        n_cols = len(col_vals)
    else:
        # Otherwise create a square grid to plot all parameter configurations
        n_rows, n_cols = grid_shape(len(parameters) * len(parameters[0]))

    # Before any processing, pull out parameter labels from input dictionaries
    parameter_labels = list(parameters[0].keys())
    # Instantiate empty subplot to build from - infer aspect ratio from first image to fit neatly
    ar = images[0].shape[0] / images[0].shape[1]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, (3 * n_rows) * ar), constrained_layout=True)

    # If working with 2 input parameters, create axis-mapped 2D grid
    if parameters and len(parameters[0]) == 2:
        ## Loop through input parameter combinations
        for image, params in zip(images, parameters):
            values = list(params.values())
            i = row_vals.index(values[0])
            j = col_vals.index(values[1])

            ax = axes[i, j]
            ax.imshow(image, cmap=cmap)
            ax.axis('off')
            if i == 0:
                ax.set_title(f"{parameter_labels[1]}: {values[1]}", fontsize=12)
            if j == 0:
                ax.text(-0.1, 0.5, f"{parameter_labels[0]}: {values[0]}", fontsize=12, va="center", ha="right", transform=ax.transAxes, rotation=90)
    # Otherwise, if working with non-2D inducible parameters, write out headered image in a square grid
    else:
        for idx, (image, params) in enumerate(zip(images, parameters)):
            i, j = divmod(idx, n_cols)
            ax = axes[i, j]
            ax.imshow(image, cmap=cmap)
            ax.axis('off')

            # Make a compact label like: param1=..., param2=..., ...
            label = ", ".join(f"{k}={v}" for k, v in params.items())
            ax.set_title(label, fontsize=10)

            # Hide any unused axes
        for ax in axes.flat[len(images):]:
            ax.axis('off')

    plt.savefig(out_dir / (str(file_name.stem) + "_collage.png"), dpi=300)
    plt.close()
