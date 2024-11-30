import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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


def plot_training_validation_loss(training_loss, validation_loss, save_path=None):
    epochs = list(range(1, len(training_loss) + 1))

    # Create the plot
    fig = go.Figure()

    # Add training loss trace
    fig.add_trace(go.Scatter(
        x=epochs, y=training_loss,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue')
    ))

    # Add validation loss trace
    fig.add_trace(go.Scatter(
        x=epochs, y=validation_loss,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='red')
    ))

    # Update layout
    fig.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend_title='Loss Type',
        template='plotly_white'
    )

    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
        print(f"Plot saved to {save_path}")
    else:
        fig.show()
