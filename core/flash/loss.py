import jax.numpy as jnp

def Mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  return jnp.mean(jnp.square(y_true - y_pred))

def Root_mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  return jnp.sqrt(Mean_squared_error(y_true, y_pred))

def Mean_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  return jnp.mean(jnp.abs(y_true - y_pred))

def Total_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  return jnp.sum(jnp.square(y_true - y_pred)) / 2.0

def Total_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  return jnp.sum(jnp.abs(y_true - y_pred))

def L1_loss(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  return jnp.mean(jnp.abs(y_pred - y_true))

# classification loss functions

def Categorical_crossentropy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  epsilon = 1e-15
  y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
  
  return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=-1))

def Sparse_categorical_crossentropy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  epsilon = 1e-15
  y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
  
  true_class_probabilities = jnp.take_along_axis(y_pred, y_true[:, None], axis=-1).squeeze(-1)
  
  return -jnp.mean(jnp.log(true_class_probabilities))

def Binary_crossentropy(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
  epsilon = 1e-15
  y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
  
  return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))
