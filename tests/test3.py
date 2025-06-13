import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons # Using make_moons for example data

def plot_decision_boundary_for_custom_model(model, X_data: list, y_data: list, title: str = "Decision Boundary"):
    """
    Plots the decision boundary of a 2D classification model.
    This function specifically handles models that:
    1. Accept input data as a list of lists (e.g., [[f1_1, f2_1], [f1_2, f2_2], ...]).
    2. Output predictions as a list (e.g., [0, 1, 0, ...] or [0.1, 0.9, 0.2, ...]).

    Args:
        model: A trained classification model instance with a 'predict' method.
               The 'predict' method should conform to the list-in/list-out interface.
        X_data (list): A list of data points, where each inner list is [feature1, feature2].
                       (e.g., [[1.2, 0.5], [-0.3, 2.1], ...])
        y_data (list): A list of true labels corresponding to X_data.
                       (e.g., [0, 1, 0, ...])
        title (str): Title for the plot.
    """
    # 1. Convert input lists to NumPy arrays for easier manipulation with numpy/matplotlib
    X_np = np.array(X_data)
    y_np = np.array(y_data)

    if X_np.shape[1] != 2:
        print("Warning: This function is best for 2-feature data. Plotting first two features.")
        X_plot = X_np[:, :2] # Use only the first two features for plotting
    else:
        X_plot = X_np

    # 2. Define a meshgrid (grid of points) to cover the feature space
    # Determine the range of the features with a small margin
    x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
    y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5

    # Create a dense grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), # Step size determines resolution
                         np.arange(y_min, y_max, 0.02))

    # 3. Prepare grid points for your custom model's 'predict' method
    # Flatten the grid points from xx, yy into a 2D array of (N, 2)
    grid_points_flat_np = np.c_[xx.ravel(), yy.ravel()]
    
    # Convert this NumPy array into a list of lists, as your model expects
    grid_points_for_model = grid_points_flat_np.tolist()

    # 4. Get predictions from your custom model
    # Your model's predict method should be called here
    raw_predictions_list = model.predict(grid_points_for_model)

    # 5. Convert predictions list back to a NumPy array for processing with Matplotlib
    raw_predictions_np = np.array(raw_predictions_list)

    # 6. Interpret model output (probabilities vs. hard labels) and convert to integer class labels
    Z = None
    unique_true_classes = np.unique(y_np)
    
    if raw_predictions_np.ndim == 1:
        # Case A: Binary classification with single probability output (e.g., [0.1, 0.9, 0.4])
        if np.issubdtype(raw_predictions_np.dtype, np.floating) and np.all((raw_predictions_np >= 0) & (raw_predictions_np <= 1)):
            Z = (raw_predictions_np > 0.5).astype(int) # Threshold probabilities at 0.5
        # Case B: Already hard labels for binary (e.g., [0, 1, 0])
        else:
            Z = raw_predictions_np.astype(int)
    elif raw_predictions_np.ndim == 2:
        # Case C: Multi-class probabilities (e.g., [[0.1,0.8,0.1],[0.7,0.2,0.1]])
        if np.issubdtype(raw_predictions_np.dtype, np.floating) and np.all((raw_predictions_np >= 0) & (raw_predictions_np <= 1)):
            Z = np.argmax(raw_predictions_np, axis=1) # Get the index of the highest probability (the class label)
        # Case D: Already one-hot encoded or multi-output hard labels (less common for this plotting type)
        # For simplicity, we'll assume argmax is the right way if 2D float output
        else:
             Z = np.argmax(raw_predictions_np, axis=1) # This handles common cases like one-hot hard labels too
    else:
        raise ValueError("Model output format not recognized. Expected 1D or 2D array for probabilities or hard labels.")
    
    # Reshape the predictions back to the grid shape
    Z = Z.reshape(xx.shape)

    # 7. Plot the contour/decision regions
    plt.figure(figsize=(9, 7))

    # Choose colormap based on number of unique classes
    num_classes = len(unique_true_classes)
    if num_classes <= 2:
        cmap_regions = plt.cm.RdYlBu # Good for binary
        cmap_points = plt.cm.RdYlBu
    else:
        # For multi-class, use a colormap that provides distinct colors for each class
        cmap_regions = plt.cm.get_cmap('viridis', num_classes)
        cmap_points = plt.cm.get_cmap('viridis', num_classes) # Or 'tab10', 'Set1' etc.
    
    # Define levels to ensure clear boundaries for integer classes
    levels = np.arange(np.min(Z), np.max(Z) + 2) - 0.5 # E.g., for classes 0,1,2, levels will be -0.5, 0.5, 1.5, 2.5

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_regions, levels=levels)

    # 8. Overlay the original data points
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_np, cmap=cmap_points,
                edgecolors='k', s=30, label="True Labels")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # Add a colorbar for clarity, especially for multi-class
    cbar = plt.colorbar(ticks=unique_true_classes, label="Predicted Class")
    cbar.ax.set_yticklabels([str(int(c)) for c in unique_true_classes]) # Ensure integer labels on colorbar

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# --- Example of a Custom Model with List Input/Output Interface ---
class MyDummyCustomModel:
    """
    A placeholder for your actual pure Python neural network.
    It simulates a simple binary classifier for 2D data.
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # In a real model, you'd have trained weights and biases here

    def predict(self, input_list_of_lists: list) -> list:
        """
        Predicts probabilities for binary classification or class labels.
        Accepts a list of lists (data points) and returns a list of outputs.
        """
        predictions = []
        for x1, x2 in input_list_of_lists:
            # This is where your actual model's forward pass would go.
            # For this dummy, let's create a simple non-linear boundary (e.g., a circle)
            # Example: Predict 1 if (x1^2 + x2^2) is less than some radius, else 0
            radius_squared = x1**2 + x2**2
            
            # Return probabilities (for ROC AUC, Log Loss etc.)
            # Or you could return hard labels directly if your model does that
            
            # Let's return probabilities for flexibility
            if radius_squared < 0.5: # Inside the circle, higher probability for class 1
                prob_class_1 = 0.9 - np.random.rand() * 0.1 # High prob
            else: # Outside the circle, lower probability for class 1
                prob_class_1 = 0.1 + np.random.rand() * 0.1 # Low prob
            
            predictions.append(prob_class_1)
            
        return predictions

# --- Example Usage with the Custom Model and Generated Data ---
if __name__ == "__main__":
    # 1. Generate some non-linear data (e.g., moons)
    n_samples = 500
    noise_level = 0.1
    X_data_list, y_data_list = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
    
    # Convert to standard Python lists for the function's expected input
    X_data_list = X_data_list.tolist()
    y_data_list = y_data_list.tolist()

    print(f"Generated {len(X_data_list)} samples.")
    print(f"First 5 X_data samples: {X_data_list[:5]}")
    print(f"First 5 y_data samples: {y_data_list[:5]}")

    # 2. Instantiate your custom model (assuming it's already trained or dummy)
    # For a real scenario, this would be your trained neural network model instance.
    my_model = MyDummyCustomModel()

    # 3. Plot the decision boundary using the new function
    plot_decision_boundary_for_custom_model(
        model=my_model,
        X_data=X_data_list,
        y_data=y_data_list,
        title="Decision Boundary of Custom Model (Moons Data)"
    )

    