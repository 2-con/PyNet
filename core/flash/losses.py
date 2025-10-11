import jax

import jax.numpy as jnp

class Mean_squared_error:
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(y_true - y_pred))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return 2 * (y_pred - y_true) / y_true.size

class Root_mean_squared_error:
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(Mean_squared_error.forward(y_true, y_pred))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    mse_grad = Mean_squared_error.backward(y_true, y_pred)
    mse = Mean_squared_error.forward(y_true, y_pred)
    return mse_grad / (2 * jnp.sqrt(mse))

class Mean_absolute_error:
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(y_true - y_pred))

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sign(y_pred - y_true))

class Total_squared_error:
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.square(y_true - y_pred)) / 2.0

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return (y_pred - y_true)

class Total_absolute_error:
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.abs(y_true - y_pred))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(y_pred - y_true)

class L1_loss:
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(y_pred - y_true))

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sign(y_pred - y_true))

# classification loss functions
class Categorical_crossentropy:
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    return -jnp.mean(jnp.sum(y_true * jnp.log(y_pred), axis=-1))

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    return -y_true / y_pred

class Sparse_categorical_crossentropy:
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    true_class_probabilities = jnp.take_along_axis(y_pred, y_true[:, None], axis=-1).squeeze(-1)
    return -jnp.mean(jnp.log(true_class_probabilities))
  
  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    num_classes = y_pred.shape[-1]
    
    one_hot_labels = jax.nn.one_hot(y_true, num_classes=num_classes)
    
    return -(one_hot_labels / y_pred)

class Binary_crossentropy:
  @staticmethod
  def forward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    return -jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))

  @staticmethod
  def backward(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    epsilon = 1e-15
    y_pred = jnp.clip(y_pred, epsilon, 1.0 - epsilon)
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
