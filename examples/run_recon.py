from dicsuite.pipeline import qpi_reconstruct_batch
from pathlib import Path

# Set path to image
input_image = Path(__file__).parent / "agar_beads.tiff"
output_dir = Path(__file__).parent / "output"

# Run a basic reconstruction with defaults
qpi_reconstruct_batch(
    files_in=input_image,
    out_dir=output_dir,
    infer_from_first=False,
    contrast_adj=True
)
