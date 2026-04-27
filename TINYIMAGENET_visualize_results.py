import matplotlib.pyplot as plt
import numpy as np

def visualize_tiny_imagenet_results(model, dataset, num_samples=25):
    """
    Visualizes Tiny ImageNet results for a Keras model.
    """
    # 1. Get predictions (Keras returns probabilities, use argmax for class)
    predictions_prob = model.predict(dataset.X)
    predictions = np.argmax(predictions_prob, axis=1)
    
    # 2. Pick random indices
    indices = np.random.choice(len(dataset.Y), num_samples, replace=False)
    
    # 3. Plotting
    cols = 5
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    plt.suptitle("Tiny ImageNet: Actual vs Predicted", fontsize=16)

    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i + 1)
        
        # Tiny ImageNet is already (64, 64, 3), no need to reshape!
        img = dataset.X[idx]
        
        # If normalized (0-1), imshow works fine. If not, cast to uint8.
        plt.imshow(img)
        
        pred = predictions[idx]
        actual = dataset.Y[idx]
        
        color = 'green' if pred == actual else 'red'
        plt.title(f"P: {pred} | A: {actual}", color=color)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Usage in your main block:
# visualize_tiny_imagenet_results(model, val_dataset, num_samples=20)