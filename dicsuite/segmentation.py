from cellpose import models, plot
import matplotlib.pyplot as plt

def run_cellpose(image, model):
    # Initialize pretrained cytoplasm model (GPU-enabled)
    model = models.Cellpose(gpu=True, model_type='cyto')
    # Run segmentation
    channels = [0, 0]  # single-channel grayscale
    masks, flows, styles, diams = model.eval(image, diameter=None, channels=channels)

    # Visualize results
    fig = plt.figure(figsize=(8, 8))
    plot.show_segmentation(fig, image, masks, flows[0], channels=channels)
    plt.show()