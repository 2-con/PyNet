# import tensorflow as tf
# from tensorflow.keras import layers, Model

# def create_unconventional_cnn_model(input_shape=(32, 32, 3)):
#   """
#   Creates a TensorFlow Keras model that approximates the user's
#   custom CNN architecture with spatial concatenation of single-channel outputs.

#   Args:
#       input_shape (tuple): The shape of the input data (e.g., (32, 32, 3) for CIFAR-10).
#                             Assuming height, width, and then channels.
#   Returns:
#       tf.keras.Model: The constructed Keras model.
#   """
#   inputs = tf.keras.Input(shape=input_shape)

#   # --- First "Convolutional Block" ---
#   # Simulates 5 net.Convolution layers in parallel, each producing 1 channel.
#   # Each Conv2D operates on the *full input* (e.g., 3 channels for an image)
#   # but outputs only 1 channel.
#   conv_outputs_branch1 = []
#   for i in range(5):
#     # We give each parallel Conv2D a unique name to differentiate them
#     conv_output = layers.Conv2D(
#       filters=1,  # Each net.Convolution produces 1 output channel
#       kernel_size=(4, 4),
#       activation='elu',
#       padding='valid',  # Default padding, reduces spatial dimensions
#       name=f'conv1_branch_{i+1}'
#     )(inputs)
#     conv_outputs_branch1.append(conv_output)

#   # Concatenate the 5 single-channel outputs spatially along the width (axis=2)
#   # The result is a single feature map with an expanded width.
#   # Shape: (Batch, H_after_conv, W_after_conv * 5, 1)
#   merged_output_block1 = layers.concatenate(conv_outputs_branch1, axis=2, name='merge_conv1_spatial')

#   # --- First Max Pooling Layer ---
#   # Operates on the single (but wide) feature map
#   pooled_output_block1 = layers.MaxPooling2D(
#     pool_size=(2, 2),
#     name='maxpool1'
#   )(merged_output_block1)

#   # --- Second "Convolutional Block" ---
#   # Simulates 5 more net.Convolution layers in parallel.
#   # Each operates on the *single-channel output* of the previous pooling layer.
#   conv_outputs_branch2 = []
#   for i in range(5):
#     conv_output = layers.Conv2D(
#       filters=1,  # Still producing 1 output channel per "branch"
#       kernel_size=(4, 4),
#       activation='elu',
#       padding='valid',
#       name=f'conv2_branch_{i+1}'
#     )(pooled_output_block1) # Input is the single-channel pooled map
#     conv_outputs_branch2.append(conv_output)

#   # Concatenate the 5 single-channel outputs spatially along the width (axis=2) again
#   # Shape: (Batch, H_after_conv2, W_after_conv2 * 5, 1)
#   merged_output_block2 = layers.concatenate(conv_outputs_branch2, axis=2, name='merge_conv2_spatial')

#   # --- Second Max Pooling Layer ---
#   # Operates on the single (but wide) feature map
#   pooled_output_block2 = layers.MaxPooling2D(
#     pool_size=(2, 2),
#     name='maxpool2'
#   )(merged_output_block2)

#   # --- Flatten Layer ---
#   # Converts the 2D feature map into a 1D vector
#   flattened_output = layers.Flatten(name='flatten')(pooled_output_block2)

#   # --- Dense Layers ---
#   dense_layer_1 = layers.Dense(64, activation='elu', name='dense_64')(flattened_output)
#   dense_layer_2 = layers.Dense(32, activation='elu', name='dense_32')(dense_layer_1)
#   # Final classification layer
#   outputs = layers.Dense(10, activation='softmax', name='output_softmax')(dense_layer_2)

#   # Create the Keras Model
#   model = Model(inputs=inputs, outputs=outputs, name="unconventional_cnn_model")
#   return model

# # --- Example Usage ---
# # Assuming common image input shape like CIFAR-10
# input_image_shape = (28, 28, 1) # Height, Width, Channels (e.g., RGB)

# # Create the model
# model = create_unconventional_cnn_model(input_image_shape)

# # Print a summary of the model's layers and output shapes
# model.summary()

# # You can now compile and train this model:
# model.compile(
#   optimizer='adam',
#   loss='categorical_crossentropy',
#   metrics=['accuracy']
# )

# # Example of dummy data (replace with your actual data)
# import numpy as np

# import sys
# import os
# current_script_dir = os.path.dirname(__file__)
# pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
# sys.path.append(pynet_root_dir)

# from datasets.image import mnist

# train_images, train_labels, test_images, test_labels = mnist(one_hot=True, normalized=True).load()

# # Example of training (using your original patience and validation split)
# history = model.fit(
#   np.array(train_images[:400]),
#   np.array(train_labels[:400]),
#   epochs=250,
#   batch_size=10,
#   validation_split=0.25,
#   callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_accuracy', restore_best_weights=True)],
#   verbose=1 # Changed from 6 to 1 for standard TensorFlow output
# )

import sys
import os
current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

import numpy as np
import matplotlib.pyplot as plt
from tools.scaler import argmax

def plot_DBR(model, X_data, y_data, epoch, fig=None, ax=None, title_prefix="Decision Boundary"):
    """
    Plots the decision boundary of a 2D binary classification model in real-time.

    Args:
        model: The pure Python model instance. Must have a `predict_proba` method
               that takes (N, 2) array and returns (N,) array of probabilities for class 1.
        X_data (np.ndarray): The 2D input features (shape: n_samples, 2).
        y_data (np.ndarray): The binary target labels (shape: n_samples,).
        epoch (int): The current training epoch.
        plot_interval (int): Plot only every 'plot_interval' epochs.
                             Set to 1 to plot every epoch.
        fig (matplotlib.figure.Figure, optional): Existing Figure object to update.
                                                  If None, a new one is created.
        ax (matplotlib.axes.Axes, optional): Existing Axes object to update.
                                             If None, a new one is created.
        title_prefix (str): Prefix for the plot title.
    """

    # # Create figure and axes if not provided (for the first plot)
    # if fig is None or ax is None:
    #     fig, ax = plt.subplots(figsize=(8, 6))

    ax.clear() # Clear the previous plot content for update

    # Convert input lists to NumPy arrays for matplotlib plotting and calculations
    X_data_np = np.array(X_data)
    y_data_np = np.array(y_data)

    # Define the bounds of the plot based on data, with a small margin
    x_min, x_max = X_data_np[:, 0].min() - 0.5, X_data_np[:, 0].max() + 0.5
    y_min, y_max = X_data_np[:, 1].min() - 0.5, X_data_np[:, 1].max() + 0.5

    # Create a meshgrid (NumPy array) for the decision regions
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points_np = np.c_[xx.ravel(), yy.ravel()]

    # --- Crucial Adaptation for your model: Process points one by one ---
    grid_points_list = grid_points_np.tolist() # Convert batch to list of single points
    
    probabilities_list = []
    for single_point_list in grid_points_list:
        # Call model.push for EACH individual point
        proba = argmax(model.push(single_point_list)) 
        probabilities_list.append(proba)
    # --- End of Adaptation ---

    # Convert the list of probabilities back to a NumPy array for plotting with matplotlib
    Z_np = np.array(probabilities_list) 
    Z = Z_np.reshape(xx.shape)

    # Plot the decision regions using contourf
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.RdBu, alpha=0.6)

    # Plot the actual data points
    # Use y_data_np directly with cmap for correct color mapping of 0s and 1s
    COLORS = []
     
    for C in y_data:
      if argmax(C) == 0:
        COLORS.append('r')
      if argmax(C) == 1:
        COLORS.append('g')
      if argmax(C) == 2:
        COLORS.append('b')
      
      
    scatter = ax.scatter(X_data_np[:, 0], X_data_np[:, 1], c=COLORS, cmap=plt.cm.coolwarm, edgecolors='k', s=20)

    # Add labels and title
    ax.set_title(f"{title_prefix} (Epoch {epoch})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Draw the canvas and pause to allow the plot to update
    fig.canvas.draw()
    plt.pause(0.0001) # Small pause to allow GUI to update
