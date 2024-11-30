import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_sample(image, pred, target, sample_id):
    # Ensure image is within the valid range
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)

    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    # Create the output folder if it doesn't exist
    output_dir = "/content/drive/MyDrive/ColabNotebooks/lab4/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Plot and save the visualization
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Input Image")
    ax[1].imshow(pred, cmap='jet')
    ax[1].set_title("Predicted Segmentation")
    ax[2].imshow(target, cmap='jet')
    ax[2].set_title("Ground Truth")

    # Save the figure to the output directory
    output_path = os.path.join(output_dir, f"sample_{sample_id}.png")
    plt.savefig(output_path)
    plt.close(fig)  # Close the figure to free memory

    print(f"Saved visualization for sample {sample_id} to {output_path}")


