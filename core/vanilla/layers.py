import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np

from tools import arraytools, scaler
from system.defaults import parametric_alpha_default, parametric_beta_default
from system.config import *

from core.vanilla import activation as Activation
from core.vanilla import initialization as Initialization
from core.vanilla.datafield import Datacontainer as dc

class Key:

  ACTIVATION = {
    # rectifiers
    'relu': Activation.ReLU,
    'softplus': Activation.Softplus,
    'mish': Activation.Mish,
    'swish': Activation.Swish,
    'leaky relu': Activation.Leaky_ReLU,
    'gelu': Activation.GELU,
    'reeu': Activation.ReEU,
    'none': Activation.Linear,
    'retanh': Activation.ReTanh,

    # normalization functions
    'binary step': Activation.Binary_step,
    'softsign': Activation.Softsign,
    'sigmoid': Activation.Sigmoid,
    'tanh': Activation.Tanh,
    
    # parametric functions
    'elu': Activation.ELU,
    'selu': Activation.SELU,
    'prelu': Activation.PReLU,
    'silu': Activation.SiLU
  }

  SCALER = {
    'standard scaler': scaler.standard_scaler,
    'min max scaler': scaler.min_max_scaler,
    'max abs scaler': scaler.max_abs_scaler,
    'robust scaler': scaler.robust_scaler,
    'softmax': scaler.softmax,
  }

  INITIALIZATION = {
    'xavier uniform in': Initialization.Xavier_uniform_in,
    'xavier uniform out': Initialization.Xavier_uniform_out,
    'he uniform': Initialization.He_uniform,
    'glorot uniform': Initialization.Glorot_uniform,
    'lecun uniform': Initialization.Lecun_uniform,
    
    'he normal': Initialization.He_normal,
    'glorot normal': Initialization.Glorot_normal,
    'lecun normal': Initialization.Glorot_normal,
    
    'default': Initialization.Default,
    'none': Initialization.Default,
  }

class Datacontainer(dc):
  def __init__(self, data, *args, **kwargs):
    super().__init__(data, *args, **kwargs)
    self.parallel = kwargs.get('parallel', False)

class Convolution:
  def __init__(self, kernel:tuple, channels:int, activation:str, **kwargs):
    """
    Convolution
    -----
      Convolution layer with valid padding, accepts and returns 2D arrays.
    -----
    Args
    -----
    - kernel                (width, height) : the kernel to apply, automatically generated
    - channels              (int)           : the amount of channels to output
    - activation            (str)           : the activation function
    - (Optional) bias       (bool)          : weither to use bias, defaults to True
    - (Optional) learnable  (bool)          : whether or not the kernel is learnable, defaults to True
    - (Optional) weights    (2D array)      : the kernel to apply, in case the kernel is not learnable
    - (Optional) name       (str)           : the name of the layer
    -----
    Activation functions
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - TANDIP

    Normalization functions
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric functions
    - ELU
    - SELU
    - PReLU
    - SiLU
    """
    self.kernel_shape = kernel
    self.channels = channels
    
    self.parametric_hyperparameters = []
    
    self.activation = activation.lower()
    self.learnable = kwargs.get('learnable', True)
    self.use_bias = kwargs.get('bias', True)
    self.bias = [0]
    self.name = kwargs.get('name', 'convolution')
    self.input_shape = 0

  def reshape_input_shape(self, input_shape):
    self.input_shape = input_shape
    
    self.alphas = arraytools.generate_array(input_shape[2], self.channels, value=parametric_alpha_default)
    self.betas = arraytools.generate_array(input_shape[2], self.channels, value=parametric_beta_default)
    self.kernels = arraytools.generate_random_array(self.kernel_shape[0], self.kernel_shape[1], input_shape[2], self.channels)
    self.bias = arraytools.generate_random_array(input_shape[2], self.channels) if self.use_bias else arraytools.generate_array(input_shape[2], self.channels)
    
  def apply(self, multichannel_input):
    
    multichannel_ans = []
    
    if len(arraytools.shape(multichannel_input)) < 3:
      multichannel_input = [multichannel_input]
    
    for channel, bias_sublist, alpha_sublist, beta_sublist in zip(self.kernels, self.bias, self.alphas, self.betas):
      
      raw_ans = []
      
      for kernel, input, bias, alpha, beta in zip(channel, multichannel_input, bias_sublist, alpha_sublist, beta_sublist):
        
        m_rows = len(input)
        m_cols = len(input[0])
        k_rows = len(kernel)
        k_cols = len(kernel[0])
        
        # calculate the dimensions of the output matrix
        output_rows = (m_rows - k_rows) + 1
        output_cols = (m_cols - k_cols) + 1
        
        # initialize output matrix
        output = [[0] * output_cols for _ in range(output_rows)]
        
        # correlate (correct terminology)
        for i in range(output_rows):
          for j in range(output_cols):
            dot_product = 0
          
            for ki in range(k_rows):
              for kj in range(k_cols):
                
                dot_product += input[i + ki][j + kj] * kernel[ki][kj]
        
            output[i][j] = Key.ACTIVATION[self.activation](dot_product, alpha=alpha, beta=beta) + bias 

        raw_ans.append(output)
      
      multichannel_ans.append(arraytools.total(*raw_ans))
    
    return multichannel_ans

  def get_weighted_sum(self, multichannel_input):
    
    multichannel_ans = []
    
    if len(arraytools.shape(multichannel_input)) < 3:
      multichannel_input = [multichannel_input]
    
    for channel, bias_sublist in zip(self.kernels, self.bias):
      
      raw_ans = []
      
      for kernel, input, bias in zip(channel, multichannel_input, bias_sublist):
        
        m_rows = len(input)
        m_cols = len(input[0])
        k_rows = len(kernel)
        k_cols = len(kernel[0])
        
        # Calculate the dimensions of the output matrix
        output_rows = (m_rows - k_rows) + 1
        output_cols = (m_cols - k_cols) + 1
        
        # Initialize the output matrix with zeros
        output = [[0] * output_cols for _ in range(output_rows)]
        
        # Perform convolution
        for i in range(output_rows):
          for j in range(output_cols):
            dot_product = 0
          
            for ki in range(k_rows):
              for kj in range(k_cols):
                dot_product += input[i + ki][j + kj] * kernel[ki][kj]
        
            output[i][j] = dot_product + bias 

        raw_ans.append(output)
      
      multichannel_ans.append(arraytools.total(*raw_ans))
      
    return multichannel_ans

class Convolution2D:
  def __init__(self, kernel, activation, **kwargs):
    """
    Convolution
    -----
      Convolution layer with valid padding, accepts and returns 2D arrays.
    -----
    Args
    -----
    - kernel                (width, height) : the kernel to apply, automatically generated
    - activation            (string)        : the activation function
    - (Optional) bias       (bool)          : weither to use bias, defaults to True
    - (Optional) learnable  (boolean)       : whether or not the kernel is learnable, defaults to True
    - (Optional) weights    (2D array)      : the kernel to apply, in case the kernel is not learnable
    - (Optional) name       (string)        : the name of the layer
    -----
    Activation functions
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - TANDIP

    Normalization functions
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric functions
    - ELU
    - SELU
    - PReLU
    - SiLU
    
    Parametric functions
    - ELU
    - SELU
    - PReLU
    - SiLU
    """
    self.kernel = arraytools.generate_random_array(kernel[0], kernel[1])
    self.alpha = parametric_alpha_default
    self.beta = parametric_beta_default
    self.activation = activation.lower()
    self.learnable = kwargs.get('learnable', True)
    self.use_bias = kwargs.get('bias', True)
    self.bias = kwargs.get('bias_value', 0)
    self.name = kwargs.get('name', 'convolution')
    self.weights = kwargs.get('weights', self.kernel)
    self.input_shape = 0
    self.kernel = self.weights

  def apply(self, input):
    self.input_shape = arraytools.shape(input)
    answer = []
    
    m_rows = len(input)
    m_cols = len(input[0])
    k_rows = len(self.kernel)
    k_cols = len(self.kernel[0])
    
    # Calculate the dimensions of the output matrix
    output_rows = (m_rows - k_rows) + 1
    output_cols = (m_cols - k_cols) + 1
    
    # Initialize the output matrix with zeros
    output = [[0] * output_cols for _ in range(output_rows)]
    
    # Perform convolution
    for i in range(output_rows):
      for j in range(output_cols):
        dot_product = 0
      
        for ki in range(k_rows):
          for kj in range(k_cols):
            dot_product += input[i + ki][j + kj] * self.kernel[ki][kj]
    
        output[i][j] = Key.ACTIVATION[self.activation](dot_product, alpha=self.alpha, beta=self.beta) + self.bias 
    
    return output

  def get_weighted_sum(self, input):
    answer = []
    
    m_rows = len(input)
    m_cols = len(input[0])
    k_rows = len(self.kernel)
    k_cols = len(self.kernel[0])
    
    # Calculate the dimensions of the output matrix
    output_rows = (m_rows - k_rows) + 1
    output_cols = (m_cols - k_cols) + 1
    
    # Initialize the output matrix with zeros
    output = [[0] * output_cols for _ in range(output_rows)]
    
    # Perform convolution
    for i in range(output_rows):
      for j in range(output_cols):
        dot_product = 0
      
        for ki in range(k_rows):
          for kj in range(k_cols):
            dot_product += input[i + ki][j + kj] * self.kernel[ki][kj]
    
        output[i][j] = dot_product + self.bias 
    
    return output

class Dense:
  def __init__(self, neurons, activation, **kwargs):
    """
    Dense (fully connected layer)
    -----
      Fully connected perceptron layer, previously reffered as 'feedforward layer', accepts and returns 1D arrays.
    -----
    Args
    -----
    - neurons                   (int)     : the number of neurons in the layer
    - activation                (string)  : the activation function
    - (Optional) initialization (string)  : intialization of the weights, defaults to Glorot uniform
    - (Optional) learnable      (boolean) : whether or not the layer is learnable, defaults to True
    - (Optional) name           (string)  : the name of the layer
    -----
    Activation functions
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - TANDIP

    Normalization functions
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric functions
    - ELU
    - SELU
    - PReLU
    - SiLU
    
    Initialization
    - Xavier uniform in
    - Xavier uniform out
    - He uniform
    - Glorot uniform
    - LeCun uniform
    - He normal
    - Glorot normal
    - LeCun normal
    - Default
    - None
    """
    self.output_shape = neurons
    self.activation = activation.lower()
    self.input_shape = kwargs.get('input_shape', 0)
    
    if activation.lower() in rectifiers:
      default_initializer = 'he normal'
    elif activation.lower() in normalization:
      default_initializer = 'glorot normal'
    else:
      raise ValueError(f"Unknown activation function: '{activation}'")
    
    self.initialization = kwargs.get('initialization', default_initializer).lower()

    neuron = {
      'weights': [0],
      'bias': 0
      }

    self.neurons = [neuron for _ in range(neurons)]

    self.learnable = kwargs.get('learnable', True)
    self.name = kwargs.get('name', 'dense')

  def reshape_input_shape(self, input_shape, output_shape):
    
    self.neurons = [
      {
      'weights': [Key.INITIALIZATION[self.initialization](input_shape, output_shape) for _ in range(input_shape)],
      'bias': Key.INITIALIZATION[self.initialization](input_shape, output_shape),
      'alpha': parametric_alpha_default,
      'beta': parametric_beta_default
      }
      for _ in range(self.output_shape)
    ]

  def apply(self, input: list):
    self.input = input
    answer = []

    if type(input) != list:
      raise TypeError("input must be a 1D array list")
    if type(input[0]) not in (int, float):
      raise TypeError("input must be a 1D array list \nuse the built-in 'Flatten' layer before a neural network layer")

    # iterate over all the neurons
    answer = [
    Key.ACTIVATION[self.activation](
      sum(input_val * weight_val for input_val, weight_val in zip(input, _neuron['weights'])) + _neuron['bias'],
      alpha=_neuron['alpha'],
      beta=_neuron['beta']
    )
    for _neuron in self.neurons
]
    return answer

  def get_weighted_sum(self, input: list):
    self.input = input[:]
    answer = []

    if type(input) != list:
      raise TypeError("input must be a 1D array list")
    if type(input[0]) not in (int, float):
      print(input)
      raise TypeError("input must be a 1D array list \nuse the built-in 'Flatten' layer before a neural network layer")

    # iterate over all the neurons
    answer = [
    (
        sum(input_val * weight_val for input_val, weight_val in zip(input, _neuron['weights'])) + _neuron['bias']
    )
    for _neuron in self.neurons
    ]
    return answer

class Localunit:
  def __init__(self, receptive_field, activation, **kwargs):
    """
    Local Unit
    -----
      Locally connected perceptron layer, accepts and returns 1D arrays.
      a constituent layer within a LCN (Locally connected networks), also known as a 'LCN' (Locally connected neurons).
    -----
    Args
    -----
    - receptive_field         (int)     : the receptive field of the layer
    - activation              (string)  : the activation function
    - (Optional) learnable    (boolean) : whether or not the layer is learnable, defaults to True
    - (Optional) name         (string)  : the name of the layer
    -----
    Activation functions
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - TANDIP

    Normalization functions
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric functions
    - ELU
    - SELU
    - PReLU
    - SiLU
    
    Initialization
    - Xavier uniform in
    - Xavier uniform out
    - He uniform
    - Glorot uniform
    - LeCun uniform
    - He normal
    - Glorot normal
    - LeCun normal
    - Default
    - None
    """
    self.receptive_field = receptive_field
    self.activation = activation.lower()
    self.name = kwargs.get('name', 'local unit')
    self.learnable = kwargs.get('learnable', True)
    
    if activation.lower() in parametric_rectifiers or activation.lower() in static_rectifiers:
      default_initializer = 'he normal'
    elif activation.lower() in normalization_functions or activation.lower() in parametric_normalization_functions:
      default_initializer = 'glorot normal'
    else:
      raise ValueError(f"Unknown activation function: '{activation}'")
    
    self.initialization = kwargs.get('initialization', default_initializer).lower()

    self.input_shape = kwargs.get('input_shape', 0)
    self.output_shape = 0 # will get calibrated soon

    neuron = {
      'weights': [0],
      'bias': 0
      }

    self.neurons = [neuron] # will be calibrated soon

    self.learnable = kwargs.get('learnable', True)
    self.name = kwargs.get('name', 'unnamed_network')

  def reshape_input_shape(self, input_shape, next_shape):

    self.output_shape = input_shape - (self.receptive_field - 1)
    
    self.neurons = [
      {
      'weights': [Key.INITIALIZATION[self.initialization](input_shape, next_shape) for _ in range(self.receptive_field)],
      'bias': Key.INITIALIZATION[self.initialization](input_shape, next_shape),
      'alpha': parametric_alpha_default,
      'beta': parametric_beta_default
      }
      for _ in range(self.output_shape)
    ]

  def apply(self, input: list):
    self.input = input
    answer = []

    if type(input) != list:
      raise TypeError("input must be a 1D array list")
    if type(input[0]) not in (int, float):
      raise TypeError("input must be a 1D array list \nuse the built-in 'Flatten' layer before a neural network layer")

    # iterate over all the neurons
    for a in range((len(self.neurons) - self.receptive_field) + 1):
      dot_product = 0

      # iterate over the input
      for b in range(self.receptive_field):

        dot_product += input[a + b] * self.neurons[a]['weights'][b]

      answer.append(Key.ACTIVATION[self.activation](
        dot_product + self.neurons[a]['bias'], 
        alpha=self.neurons[a]['alpha'], 
        beta=self.neurons[a]['beta']
        )
      )
    return answer

  def get_weighted_sum(self, input: list):
    self.input = input
    answer = []

    if type(input) != list:
      raise TypeError("input must be a 1D array list")
    if type(input[0]) not in (int, float):
      raise TypeError("input must be a 1D array list \nuse the built-in 'Flatten' layer before a neural network layer")

    # iterate over all the neurons
    for a in range((len(self.neurons) - self.receptive_field) + 1):
      dot_product = 0

      # iterate over the input
      for b in range(self.receptive_field):

        dot_product += input[a + b] * self.neurons[a]['weights'][b]

      answer.append(dot_product + self.neurons[a]['bias'])
    return answer

class Recurrent:
  def __init__(self, activation, **kwargs):
    """
    Recurrent Unit
    -----
      Primary block within RNNs (Recurrent Neural Networks). it is only compatible with other Recurrent Unit.
    -----
    Args
    -----
    - activation           (string)  : the activation function to use for the attention layer
    - (Optional) input     (boolean) : accept an input during propagation, on by default
    - (Optional) output    (boolean) : return anything during propagation, on by default
    - (Optional) learnable (boolean) : whether or not to learn, on by default
    - (Optional) name      (string)  : the name of the layer
    """
    self.activation = activation.lower()
    
    # activation parameters
    self.alpha = parametric_alpha_default
    self.beta = parametric_beta_default
    
    self.accept_input = kwargs.get('input', True)
    self.return_output = kwargs.get('output', True)
    
    self.name = kwargs.get('name', 'recurrent')
    self.learnable = kwargs.get('learnable', True)
    
    self.input_weight = random.uniform(0.1, 1)
    self.carry_weight = random.uniform(0.1, 1)
    self.bias = random.uniform(-0.5, 0.5)
  
  def apply(self, input, carry):
    
    return Key.ACTIVATION[self.activation]((input * self.input_weight) + (carry * self.carry_weight) + self.bias, alpha=self.alpha, beta=self.beta)

  def get_weighted_sum(self, input, carry):
    
    return (input * self.input_weight) + (carry * self.carry_weight) + self.bias

class LSTM:
  def __init__(self, **kwargs):
    """
    Long Short Term Memory (LSTM)
    -----
      Simmilar to the recurrent unit, but it has a long term memory and this, it is only compatible with other LSTMs.
      it is not copatible with Gated Recurrent Unit (GRU) or Recurrent Unit (RNN) since they only have 1 state.
    -----
    Args
    -----
    - (Optional) version   (boolean) : alters the version of the LSTM, 'Standard' by default
    - (Optional) input     (boolean) : accept an input during propagation, on by default
    - (Optional) output    (boolean) : return anything during propagation, on by default
    - (Optional) learnable (boolean) : whether or not to learn, on by default
    - (Optional) name      (string)  : the name of the layer
    -----
      Versions
    -----
      - Standard (default) : the standard LSTM, the output is not scaled and thus, have a range of -1 to 1
      - Statquest : an LSTM architechture that is proposed by Statquest, it scales the 
                    output by independent weights, allowing it to have a range of -infinity to infinity
    """
    
    self.name = kwargs.get('name', 'LSTM')
    self.accept_input = kwargs.get('input', True)
    self.return_output = kwargs.get('output', True)
    self.learnable = kwargs.get('learnable', True)
    self.version = kwargs.get('version', 'standard').lower()
    
    STW_RANGE  = 1.0
    INP_RANGE  = 1
    BIAS_RANGE = 0
    EXW_RANGE  = 1.0
    
    self.short_term_weights = [random.uniform(-STW_RANGE,  STW_RANGE) ] * 4
    self.input_weights      = [random.uniform(-INP_RANGE,  INP_RANGE) ] * 4
    self.biases             = [random.uniform(-BIAS_RANGE, BIAS_RANGE)] * 4
    self.extra_weights      = [random.uniform(-EXW_RANGE,  EXW_RANGE) ] * 3 if self.version == 'statquest' else [1, 1, 1]
  
  def apply(self, input, long_term, short_term):
    
    # calculate the merged state
    
    weighted_input = [input * self.input_weights[i] for i in range(4)]
    weighted_short_term = [short_term * self.short_term_weights[i] for i in range(4)]
    
    merged_state = [weighted_input[i] + weighted_short_term[i] + self.biases[i] for i in range(len(weighted_input))]

    # process the forget gate
    gated_long_term = Key.ACTIVATION["sigmoid"](merged_state[0]) * long_term
    
    # process the input gate
    final_long_term = (Key.ACTIVATION["sigmoid"](merged_state[1]) * Key.ACTIVATION["tanh"](merged_state[2])) + gated_long_term
    
    # process the output gate
    final_short_term = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term) * self.extra_weights[0] * self.extra_weights[1] * self.extra_weights[2]

    # return the final long term and short term states
    return final_long_term, final_short_term
  
  def get_weighted_sum(self, input, long_term, short_term):
    # calculate the merged state
    
    weighted_input = [input * self.input_weights[i] for i in range(4)]
    weighted_short_term = [short_term * self.short_term_weights[i] for i in range(4)]
    
    merged_state = [weighted_input[i] + weighted_short_term[i] + self.biases[i] for i in range(len(weighted_input))]

    # process the forget gate
    gated_long_term = Key.ACTIVATION["sigmoid"](merged_state[0]) * long_term
    
    # process the input gate
    final_long_term = (Key.ACTIVATION["sigmoid"](merged_state[1]) * Key.ACTIVATION["tanh"](merged_state[2])) + gated_long_term
    
    # process the output gate
    final_short_term = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term) * self.extra_weights[0] * self.extra_weights[1] * self.extra_weights[2]
    
    # return the final long term and short term states
    return long_term, short_term, final_long_term, final_short_term, merged_state, gated_long_term, weighted_input, weighted_short_term
  
class GRU:
  def __init__(self, **kwargs):
    """
    Gated Recurrent Unit (GRU)
    -----
      Simmilar to recurrent units, just more intricate and thus, is also compatible with the recurrent unit.
      it is not copatible with Long Short Term Memory (LSTM) cells.
      
      unlike other recurrent units, GRU cells output a range from -1 to 1
    -----
    Args
    -----
    - (Optional) input     (boolean) : accept an input during propagation, on by default
    - (Optional) output    (boolean) : return anything during propagation, on by default
    - (Optional) learnable (boolean) : whether or not to learn, on by default
    - (Optional) name      (string)  : the name of the layer
    """
    
    self.name = kwargs.get('name', 'recurrent')
    self.accept_input = kwargs.get('input', True)
    self.return_output = kwargs.get('output', True)
    self.learnable = kwargs.get('learnable', True)
    
    STW_RANGE  = 1.0
    INP_RANGE  = 0.75
    BIAS_RANGE = 0
    
    self.carry_weights = [random.uniform(-STW_RANGE,  STW_RANGE) ] * 3
    self.input_weights = [random.uniform(-INP_RANGE,  INP_RANGE) ] * 3
    self.biases        = [random.uniform(-BIAS_RANGE, BIAS_RANGE)] * 3
  
  def apply(self, input, carry):
    
    # merging the input and carry states
    
    weighted_input = [input * self.input_weights[i] for i in range(3)]
    weighted_carry = [carry * self.carry_weights[i] for i in range(2)]
    merged_state = [weighted_input[i] + weighted_carry[i] + self.biases[i] for i in range(2)]
    
    # calculating the final merged state
    weighted_carry.append(Key.ACTIVATION["sigmoid"](merged_state[0]) * carry * self.carry_weights[2])
    merged_state.append(weighted_input[2] + weighted_carry[2] + self.biases[2])
    
    # processes
    
    gated_carry = Key.ACTIVATION["sigmoid"](merged_state[1]) * carry
    
    final_output = Key.ACTIVATION["tanh"](merged_state[2]) * ( 1 - Key.ACTIVATION["sigmoid"](merged_state[1]) ) + gated_carry
    
    return final_output

  def get_weighted_sum(self, input, carry):
    # merging the input and carry states
    
    weighted_input = [input * self.input_weights[i] for i in range(3)]
    weighted_carry = [carry * self.carry_weights[i] for i in range(2)]
    merged_state = [weighted_input[i] + weighted_carry[i] + self.biases[i] for i in range(2)]
    
    # calculating the final merged state
    weighted_carry.append(Key.ACTIVATION["sigmoid"](merged_state[0]) * carry * self.carry_weights[2])
    merged_state.append(weighted_input[2] + weighted_carry[2] + self.biases[2])
    
    # processes
    
    gated_carry = Key.ACTIVATION["sigmoid"](merged_state[1]) * carry
    
    final_output = Key.ACTIVATION["tanh"](merged_state[2]) * ( 1 - Key.ACTIVATION["sigmoid"](merged_state[1]) ) + gated_carry
    
    return final_output, gated_carry, merged_state, weighted_input, weighted_carry, input, carry

# Functional layers

class Maxpooling:
  def __init__(self, size, stride, **kwargs):
    """
    Meanpooling
    -----
      Scales down any 2D array by pooling, accepts and returns 2D arrays.
    -----
    Args
    -----
    - size            (int)    : the size of the pooling window 
    - stride          (int)    : the stride of the pooling window
    - (Optional) name (string) : the name of the layer
    """
    self.size = size
    self.stride = stride
    self.name = kwargs.get('name', 'maxpooling')
    self.input_size = 0

  def apply(self, input_channels):
    multichannel_ans = []
    answer = []
    self.input_size = arraytools.shape(input_channels)

    for input in input_channels:
      answer = []
      
      # iterate over all the layers
      for a in range(0, len(input), self.stride):

        layer_output = []

        # iterate over all the elements in the layer
        for b in range(0, len(input[a]), self.stride):
          pool = []

          # control the vertical
          for c in range(self.size):

            # control the horizontal
            for d in range(self.size):

              if a + c < len(input) and b + d < len(input[a]):
                pool.append(input[a+c][b+d])

          layer_output.append( max(pool) )

        answer.append(layer_output[:])
        
      multichannel_ans.append(answer)

    return multichannel_ans

class Meanpooling:
  def __init__(self, size, stride, **kwargs):
    """
    Meanpooling
    -----
      Scales down any 2D array by pooling, accepts and returns 2D arrays.
    -----
    Args
    -----
    - size            (int)  : the size of the pooling window
    - stride          (int)  : the stride of the pooling window
    - (optional) name (string) : the name of the layer
    """
    self.size = size
    self.stride = stride
    self.name = kwargs.get('name', 'meanpooling')
    self.input_size = 0

  def apply(self, input_channels):
    multichannel_ans = []
    answer = []
    self.input_size = arraytools.shape(input_channels)

    for input in input_channels:
      answer = []
      
      # iterate over all the layers
      for a in range(0, len(input), self.stride):

        layer_output = []

        # iterate over all the elements in the layer
        for b in range(0, len(input[a]), self.stride):
          pool = []

          # control the vertical
          for c in range(self.size):

            # control the horizontal
            for d in range(self.size):

              if a+c < len(input) and b+d < len(input[a]):
                pool.append(input[a+c][b+d])

          layer_output.append( sum(pool) / len(pool) )

        answer.append(layer_output[:])
      multichannel_ans.append(answer)

    return multichannel_ans

class Flatten:
  def __init__(self, **kwargs):
    """
    Flatten
    -----
      Flattens any 2D array into a 1D array, use this layer before a neural network layer (Dense layer)
    -----
    Args
    -----
    - (optional) name (string): the name of the layer
    """
    self.name = kwargs.get('name', 'flatten')
    self.neurons = [0]
    self.input_shape = 0

  def apply(self, input):
    
    self.input_shape = arraytools.shape(input)
    
    return arraytools.flatten(input)

  def set_length(self, length):
    self.neurons = [0 for _ in range(length)]

class Reshape:
  def __init__(self, width, height, **kwargs):
    """
    Reshape
    -----
      Reshapes an array, accepts and returns 2D arrays.
    -----
    Args
    -----
    - height (int)  : the height of the output
    - width (int)   : the width of the output
    - (Optional) name (string) : the name of the layer
    -----
    negative values means that the shape will be inferred, essencially counting backwards just
    like how the python range function works
    """
    self.height = height
    self.width = width
    self.name = kwargs.get('name', 'reshape')
    self.input_shape = 0

  def apply(self, input):
    self.input_shape = arraytools.shape(input)
    return arraytools.reshape(input, (self.width, self.height))

class Operation:
  def __init__(self, operation, **kwargs):
    """
    Operation
    -----
      Operational layer for functions or normalizations.
      
      Parametric functions will not be able to change their hyperparameters since it defeats an operation-only layer.
    -----
    Args
    -----
    - operation    (string) : the scaler to use
    - (Optional) name   (string) : the name of the layer
    -----
    Active Operations:

    Scalers
    - standard scaler
    - min max scaler
    - max abs scaler
    - robust scaler

    Activation functions
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - TANDIP

    Normalization functions
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    - Softmax
    
    Parametric functions
    - ELU
    - SELU
    - PReLU
    - SiLU
    """
    self.operation = operation.lower()
    self.name = kwargs.get('name', 'operation')

    self.minimum = 0
    self.maximum = 0
    self.q1 = 0
    self.q3 = 0
    self.std = 0

    self.neurons = [0]

  def reshape_input_shape(self, input_shape, output_shape):

    self.neurons = [0 for _ in range(input_shape)]

  def apply(self, input):

    if self.operation == 'dropout':
      # Placeholder for dropout mechanism
      pass

    # activation functions
    if self.operation in Key.ACTIVATION:
      return [Key.ACTIVATION[self.operation](a) for a in input]

    # if its a minmax scaler
    if self.operation == 'min max scaler':

      self.minimum = min(input)
      self.maximum = max(input)

      return scaler.min_max_scaler(input, min=self.minimum, max=self.maximum)

    # defaults to a scaler
    if self.operation in Key.SCALER:

      self.q1 = float(np.percentile(input, 25))
      self.q3 = float(np.percentile(input, 75))
      self.std = float(np.std(input))

      return Key.SCALER[self.operation](input)

  def get_weighted_sum(self, x):
    return x

class Dropout:
  def __init__(self, dropout_rate, **kwargs):
    """
    Dropout
    -----
      Decides how many information passing through it gets dropped, either as a percentage or a fixed number.
      depending on the method, it will default to independent percentages.
    -----
    Args
    -----
      dropout_rate      (float)  : the percentage of information to drop
      (Optional) method (string) : the initialization method, defaults to 'independent'
      (Optional) scaled (bool)   : wether to inversely scale activated values, default to True
      (Optional) name   (string) : the name of the layer
    ---
    Initialization methods:
     - 'independent' : each neurons is dropped independently of eachother in accordance to the dropout rate
     - 'fixed'       : a fixed number of neurons are dropped in accordance to the dropout rate
    """
    
    if dropout_rate < 0 or dropout_rate > 1:
      raise ValueError("Dropout rate must be between 0 and 1.")
    
    self.dropout_rate = dropout_rate
    self.name = kwargs.get('name', 'dropout')
    self.method = kwargs.get('method', 'independent')
    self.is_scaled = kwargs.get('scaled', True)
    self.mask = [0]
  
  def apply(self, input):
    
    scale = 1/(1 - self.dropout_rate) if self.is_scaled else 1
    
    if len(arraytools.shape(input)) == 2:
      
      if self.method == 'fixed':
        actives = [scale for _ in range( (arraytools.shape(input)[0] * arraytools.shape(input)[1]) - int((arraytools.shape(input)[0] * arraytools.shape(input)[1]) * self.dropout_rate) )]
        self.mask = [0 for _ in range((arraytools.shape(input)[0] * arraytools.shape(input)[1]) - len(actives))] + actives
        random.shuffle(self.mask)
        self.mask = arraytools.reshape(self.mask, arraytools.shape(input)[0], arraytools.shape(input)[1])
        
      elif self.method == 'independent':
        self.mask = [
          [scale if random.random() > self.dropout_rate else 0 for _ in range(len(input[0]))]
          for _ in range(len(input))]
      
      else:
        raise ValueError(f"Unknown method: '{self.method}', Method must be 'independent' or 'fixed'")
      
      return [
        [a * b for a, b in zip(row, gate)] for row, gate in zip(input, self.mask)
      ]
      
    elif len(arraytools.shape(input)) == 1:
      
      if self.method == 'fixed':
        actives = [scale for _ in range( len(input) - int(len(input) * self.dropout_rate) )]
        self.mask = [0 for _ in range(len(input) - len(actives))] + actives
        random.shuffle(self.mask)
        
      elif self.method == 'independent':
        self.mask = [scale if random.random() > self.dropout_rate else 0 for _ in range(len(input))]
      
      else:
        raise ValueError(f"Unknown method: '{self.method}', Method must be 'independent' or 'fixed'")
      
      return [a * b for a, b in zip(input, self.mask)]

    else:
      raise TypeError("input to a Dropout layer must be a 1D array list or a 2D array list")


