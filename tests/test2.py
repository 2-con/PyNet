import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from core.encoder import OneHotEncode
import numpy as np

from sklearn.datasets import make_moons, make_blobs

def plot_decision_boundary(model, X_data, y_data, step_size):
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

  x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
  y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5

  # Create a dense grid of points
  xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), # Step size determines resolution
                        np.arange(y_min, y_max, step_size))

  grid_points_flat_np = np.c_[xx.ravel(), yy.ravel()]
  grid_points_for_model = grid_points_flat_np.tolist()
  
  raw_predictions_list = [model.push(point) for point in grid_points_for_model]
  raw_predictions_np = np.array(raw_predictions_list)

  # 6. Interpret model output (probabilities vs. hard labels) and convert to integer class labels
  Z = None
  unique_true_classes = 2
  
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
  num_classes = 2
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
  COLORS = ['r' if C[0] == 0 else 'b' for C in y_data]
  plt.scatter(X_plot[:, 0], X_plot[:, 1], c=COLORS, cmap=cmap_points,
              edgecolors='k', s=30, label="True Labels")

  plt.xlabel("Feature 1")
  plt.ylabel("Feature 2")
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())

  plt.grid(True, linestyle='--', alpha=0.6)
  plt.show()

training_features, training_target = make_blobs(n_samples=200, n_features=2, centers=2, random_state=0, cluster_std=0.5)

training_features = training_features.tolist()
training_target = [ OneHotEncode(2, x)  for x in training_target.tolist()]

