import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax.numpy as jnp
import jax 
from typing import *

weights = jnp.array([
[
  [
    [-0.8607991  ,-0.8733773 ],
    [ 0.87437105 , 0.792711  ]
  ]
],
[
  [
    [-0.4012282  ,-0.33765268],
    [-0.5078237  , 0.7410147 ]
  ]
]
])

d_WS = jnp.array([
[
  [
    [ 0.73941404,  1.5208126 ],
    [ 0.8253685 , -0.45574623]
  ],
  [
    [ 0.9165013 , -0.32393005],
    [ 2.0841188 ,  0.8995146 ]
  ]
],
[
  [
    [ 2.5697803 ,  5.1935043 ],
    [ 2.797557  , -1.5548851 ]
  ],
  [
    [ 3.1394334 , -1.1328145 ],
    [ 7.1010437 ,  3.0736392 ]
  ]
]
])

strides = (1, 1)

def transposed_convolve(d_WS: jnp.ndarray, weights: jnp.ndarray, strides: Tuple) -> jnp.ndarray:
  N, C_out, H_out, W_out = d_WS.shape
  
  # 1. Dilate the incoming error signal (d_WS)
  dilated_H = H_out + (H_out - 1) * (strides[0] - 1)
  dilated_W = W_out + (W_out - 1) * (strides[1] - 1)
  
  dilated_d_WS = jnp.zeros((N, C_out, dilated_H, dilated_W), dtype=d_WS.dtype)
  
  dilated_d_WS = dilated_d_WS.at[:, :, ::strides[0], ::strides[1]].set(d_WS)
  
  flipped_weights = jnp.flip(weights, axis=(2, 3))
  
  def convolve_image(error_slice, kernel_slice):
    return jax.scipy.signal.convolve(error_slice, kernel_slice, mode='full')
  
  convolve_over_channels = jax.vmap(convolve_image, in_axes=(None, 1))

  all_convolutions = jax.vmap(convolve_over_channels, in_axes=(0, None))
  
  return jnp.sum(all_convolutions(dilated_d_WS, flipped_weights), axis=2)

upstream_gradient = transposed_convolve(d_WS, weights, strides)

print("Upstream Gradient Shape:", upstream_gradient.shape)
print("\nUpstream Gradient:")
print(upstream_gradient)
