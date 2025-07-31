from dicsuite.pipeline import qpi_reconstruct_batch
from pathlib import Path

# Set path to image
input_image = Path(__file__).parent / "agar_beads.tiff"
output_dir = Path(__file__).parent / "dic_qpi_outputs"

# Run a basic reconstruction with defaults
qpi_reconstruct_batch(
    files_in=input_image,
    out_dir=output_dir,
    smooth_in=[0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10],
    stabil_in = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10],
    infer_from_first=False,
    contrast_adj=True,
    use_gpu=True,
    write_collage=True
)