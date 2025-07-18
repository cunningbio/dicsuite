import pandas as pd
from pathlib import Path

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
        print("Shear angle registry found!")
        shear_df = pd.read_csv(log_path)

        # Ensure logs contain necessary columns and store matching registry
        if "Image_Path" not in shear_df.columns or "Shear_Angle" not in shear_df.columns:
            raise ValueError("Log file must contain 'Image_Path' and 'Shear_Angle' columns.")
        matching = shear_df[shear_df["Image_Path"] == img_path]

        # Next, check the match to ensure we have either 1 or 0 registers
        if len(matching) == 1:
            print("Shear angle found in registry! Skipping shear computation...")
            return shear_df, float(matching["Shear_Angle"].iloc[0])
        elif len(matching) > 1:
            raise ValueError("Multiple matching shear angles in registry — wipe and try again!")
        else:
            print("Shear angle not found in registry, computing...")
            return shear_df, None

    # If log doesn't exist, create a blank file to fill in once estimation is completed
    else:
        print("No shear angle registry detected. Creating logs, then computing shear angle...")
        # Create empty directory
        log_path.parent.mkdir(parents=True, exist_ok=True)
        empty_log = pd.DataFrame(columns=["Image_Path", "Shear_Angle"])
        empty_log.to_csv(log_path)
        return empty_log, None


def update_shear_log(log_df, image_path, angle):
    shear_out = pd.DataFrame([{"Name": image_path, "ShearAngle": angle}])
    # If registry is empty, just capture the new output as the full registry
    shear_out = (shear_out.copy() if log_df.empty else pd.concat([log_df, shear_out], ignore_index=True))

    return shear_out

def save_shear_log(log_df, path):
    log_df.to_csv(path / "shear_angle_log.csv")
