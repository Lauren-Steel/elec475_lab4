import os
import matplotlib.pyplot as plt

def visualize_sample(image, pred, target, sample_id):
    output_dir = 'visualizations'  # Changed from '/content/...' to a relative path
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image.permute(1, 2, 0))  # Assuming image is a normalized tensor
    ax[0].set_title('Input Image')
    ax[1].imshow(pred, cmap='gray')  # Assuming pred is a segmentation mask
    ax[1].set_title('Predicted Mask')
    ax[2].imshow(target, cmap='gray')  # Assuming target is a segmentation mask
    ax[2].set_title('Ground Truth')

    plt.savefig(os.path.join(output_dir, f'sample_{sample_id}.png'))
    plt.close()
