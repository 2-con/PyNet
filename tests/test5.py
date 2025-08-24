import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.visual import array_display
import jax.numpy as jnp
import jax

def Correlate(inputs, errors, kernel_shape, strides):
  N, C_in, H_in, W_in = inputs.shape
  _, C_out, H_out, W_out = errors.shape
  kH, kW = kernel_shape
  sH, sW = strides

  grad_weights = jnp.zeros((C_out, C_in, kH, kW))

  # Loop over the batches
  for n in range(N):
    # Loop over the output spatial dimensions
    for h_out in range(H_out):
      for w_out in range(W_out):
        # Calculate the slice for the input patch
        h_start, w_start = h_out * sH, w_out * sW
        input_patch = jax.lax.slice(
          inputs[n],
          (0, h_start, w_start),
          (C_in, h_start + kH, w_start + kW)
        )

        # Get the error for the current output position
        error_patch = jax.lax.slice(
          errors[n],
          (0, h_out, w_out),
          (C_out, h_out + 1, w_out + 1)
        ).reshape(C_out, 1, 1, 1)

        # Compute the outer product and add to the gradient
        # error_patch: (C_out, 1, 1, 1)
        # input_patch: (C_in, kH, kW)
        # The result has shape (C_out, C_in, kH, kW)
        grad_weights += error_patch * jnp.expand_dims(input_patch, axis=0)

  return grad_weights
# Test
a = jnp.array([[[[1,1],
                 [1,1]]]])  # (N=1, C_out=1, H=2, W=2)

b = jnp.array([[[[1,1],
                 [1,1]]]])  # (C_out=1, C_in=1, kH=2, kW=2)

print(Correlate(a, b, (1,1)))