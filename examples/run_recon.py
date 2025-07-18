# example usage script: examples/run_recon.py
import matplotlib.pyplot as plt
from qpi_toolkit import qpi_reconstruct

img = plt.imread("your_image.tif")
recon = qpi_reconstruct(img, use_gpu=True)
plt.imshow(recon)
plt.show()
