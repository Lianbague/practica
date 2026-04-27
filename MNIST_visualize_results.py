import matplotlib.pyplot as plt
import numpy as np

from MNIST_neural_network_training import forward_prop, get_predictions

def visualize_test_predictions(nn, num_samples=25):
    """
    Visualizes a grid of test images with Actual vs Predicted labels.
    """
    # 1. Get predictions for the entire test set
    X_test_norm = nn.X_test_raw.T / 255.
    _, _, _, a2_test = forward_prop(nn.w1, nn.b1, nn.w2, nn.b2, X_test_norm)
    predictions = get_predictions(a2_test)
    
    # 2. Pick random indices to display
    indices = np.random.choice(len(nn.Y_test_raw), num_samples, replace=False)
    
    # 3. Plotting
    cols = 5
    rows = num_samples // cols
    plt.figure(figsize=(12, 2.5 * rows))
    plt.suptitle(f"Results for Format: {nn.format.upper()}", fontsize=16)

    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        img = nn.X_test_raw[idx].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        
        pred = predictions[idx]
        actual = nn.Y_test_raw[idx]
        
        color = 'green' if pred == actual else 'red'
        plt.title(f"P: {pred} | A: {actual}", color=color)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()