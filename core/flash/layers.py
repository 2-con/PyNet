import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import typing
from system.config import *
import core.standard.functions as activations
import core.standard.initializers as initializers
import core.standard.encoders as encoders
from core.standard.layers import *

class Key:

  FUNCTION = {
    
    # normalization activations
    "sigmoid": activations.Sigmoid(),
    "tanh": activations.Tanh(),
    "binary step": activations.Binary_step(),
    "softsign": activations.Softsign(),
    "softmax": activations.Softmax(),
    
    # rectifiers
    "relu": activations.ReLU(),
    "softplus": activations.Softplus(),
    "mish": activations.Mish(),
    "swish": activations.Swish(),
    "leaky relu": activations.Leaky_ReLU(),
    "gelu": activations.GELU(),
    "identity": activations.Linear(),
    "reeu": activations.ReEU(),
    "retanh": activations.ReTanh(),
    
    # parametric activations
    'elu': activations.ELU(),
    "selu": activations.SELU(),
    "prelu": activations.PReLU(),
    "silu": activations.SiLU(),
    
    "standard scaler": activations.Standard_Scaler(),
    "min max scaler": activations.Min_Max_Scaler(),
    "max abs scaler": activations.Max_Abs_Scaler(),
    "robust scaler": activations.Robust_Scaler(),
  }
  
  ENCODER = {
    "sinusoidal positional": encoders.SinusoidalEmbedding,
    "one hot": encoders.OneHot,
    "ordinal": encoders.OrdinalEncoder
  }
  
  INITIALIZER = {
    "glorot uniform": initializers.Glorot_Uniform,
    "glorot normal": initializers.Glorot_Normal,
    "he uniform": initializers.He_Uniform,
    "he normal": initializers.He_Normal,
    "lecun uniform": initializers.Lecun_Uniform,
    "lecun normal": initializers.Lecun_Normal,
    "xavier uniform in": initializers.Xavier_Uniform_In,
    "xavier uniform out": initializers.Xavier_Uniform_Out,
    "default": initializers.Default
  }

class Dense(Dense):
  def __init__(self, neurons:int, activation:str, name:str="Null", *args, **kwargs):
    """
    Dense
    -----
      A fully connected layer that connects the previous layer to the next layer. Accepts and returns 1D arrays (excludes batch dimension), so input_shape should be of the form
      (input_size,), anything after the 1st dimention will be ignored.
    -----
    Args
    -----
    - neurons         (int)     : the number of neurons in the layer
    - activation      (string)  : the activation activation
    - (Optional) name (string)  : the name of the layer
    -----
    activation activations
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - ReTanh

    Normalization activations
    - Softmax
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric activations
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
    self.neuron_amount = neurons
    self.name = name
    
    if type(activation) == str:
      assert activation in Key.FUNCTION, f"Unknown activation: '{activation}'. Available: {list(Key.FUNCTION.keys())}"
      self.function = Key.FUNCTION[activation.lower()]
    else:
      raise ValueError("activation must be a string or an object with 'forward' and 'backward' methods.")

    self.initializer = Key.INITIALIZER['default']()
    if activation.lower() in rectifiers:
      self.initializer = Key.INITIALIZER['he normal']()
    elif activation.lower() in normalization:
      self.initializer = Key.INITIALIZER['glorot normal']()

    if 'initializer' in kwargs:
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer = Key.INITIALIZER[kwargs['initializer'].lower()]()

    self.input_size = None

class Localunit(Localunit):
  def __init__(self, receptive_field:int, activation:str, name:str="Null", *args, **kwargs):
    """
    LocalUnit (Locally Connected Layer)
    -----
      A locally connected layer that connects the previous layer to the next layer. Accepts and returns 1D arrays (excludes batch dimension), so input_shape should be of the form
      (input_size,), anything after the 1st dimention will be ignored.
    -----
    Args
    -----
    - receptive_field (int)    : the size of the receptive field for each neuron
    - activation      (string) : the activation activation
    - (Optional) name (string) : the name of the layer
    -----
    activation activations
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - ReTanh

    Normalization activations
    - Softmax
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric activations
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
    self.name = name
    
    if type(activation) == str:
      assert activation in Key.FUNCTION, f"Unknown activation: '{activation}'. Available: {list(Key.FUNCTION.keys())}"
      self.function = Key.FUNCTION[activation.lower()]
    else:
      raise ValueError("activation must be a string or an object with 'forward' and 'backward' methods.")

    self.initializer = Key.INITIALIZER['default']()
    if activation.lower() in rectifiers:
      self.initializer = Key.INITIALIZER['he normal']()
    elif activation.lower() in normalization:
      self.initializer = Key.INITIALIZER['glorot normal']()

    if 'initializer' in kwargs:
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer = Key.INITIALIZER[kwargs['initializer'].lower()]()

    self.input_size = None

class Convolution(Convolution):
  def __init__(self, kernel:tuple[int, int], channels:int, activation:str, stride:tuple[int, int], name:str="Null", *args, **kwargs):
    """
    Convolution
    -----
      Convolution that is fixed with a valid padding and no dilation. Accepts and returns 3D arrays (excludes batch dimension), so input_shape should be of the form
      (Channels, Image Height, Image Width).
    -----
    Args
    -----
    - kernel          (tuple[int, int]) : the kernel dimensions to apply, automatically generated
    - channels        (int)             : the number of channels in the kernel
    - stride          (tuple[int, int]) : the stride to apply to the kernel
    - activation      (string)          : the activation activation
    - (Optional) name (string)          : the name of the layer
    -----
    activation activations
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - ReTanh

    Normalization activations
    - Softmax
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric activations
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
    
    self.kernel = kernel
    self.channels = channels
    self.stride = stride
    self.name = name

    if type(activation) == str:
      assert activation in Key.FUNCTION, f"Unknown activation: '{activation}'. Available: {list(Key.FUNCTION.keys())}"
      self.function = Key.FUNCTION[activation.lower()]
    else:
      raise ValueError("activation must be a string or an object with 'forward' and 'backward' methods.")

    self.params = {}
    self.input_shape = None
    self.output_shape = None

    if "initializer" in kwargs:
      if kwargs["initializer"].lower() not in Key.INITIALIZER:
        raise ValueError(
          f"Unknown initializer: '{kwargs['initializer'].lower()}'. "
          f"Available: {list(Key.INITIALIZER.keys())}"
        )
      self.initializer = Key.INITIALIZER[kwargs["initializer"].lower()]()
    else:
      # He for rectifiers, Glorot for normalizers, else default
      if activation.lower() in rectifiers:
        self.initializer = Key.INITIALIZER["he normal"]()
      elif activation.lower() in normalization:
        self.initializer = Key.INITIALIZER["glorot normal"]()
      else:
        self.initializer = Key.INITIALIZER["default"]()

class Deconvolution(Deconvolution):
  def __init__(self, kernel:tuple[int, int], channels:int, activation:str, stride:tuple[int, int], name:str="Null", *args, **kwargs):
    """
    Deconvolution
    -----
      a Deconvolution layer within the context of deep learning is actually a transposed convolution. Accepts and returns 3D arrays (excludes batch dimension), so input_shape should be of the form
      (Channels, Image Height, Image Width).
    -----
    Args
    -----
    - kernel          (tuple[int, int]) : the kernel dimensions to apply, must be of the form (kernel_height, kernel_width)
    - channels        (int)             : the number of channels in the kernel
    - stride          (tuple[int, int]) : the stride to apply to the kernel
    - activation      (string)          : the activation activation
    - (Optional) name (string)          : the name of the layer
    -----
    activation activations
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - ReTanh

    Normalization activations
    - Softmax
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric activations
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
    
    self.kernel = kernel
    self.channels = channels
    self.name = name
    self.stride = stride

    if type(activation) == str:
      assert activation in Key.FUNCTION, f"Unknown activation: '{activation}'. Available: {list(Key.FUNCTION.keys())}"
      self.function = Key.FUNCTION[activation.lower()]
    else:
      raise ValueError("activation must be a string or an object with 'forward' and 'backward' methods.")

    self.params = {}
    self.input_shape = None
    self.output_shape = None

    if "initializer" in kwargs:
      if kwargs["initializer"].lower() not in Key.INITIALIZER:
        raise ValueError(
          f"Unknown initializer: '{kwargs['initializer'].lower()}'. "
          f"Available: {list(Key.INITIALIZER.keys())}"
        )
      self.initializer = Key.INITIALIZER[kwargs["initializer"].lower()]()
    else:
      # He for rectifiers, Glorot for normalizers, else default
      if activation.lower() in rectifiers:
        self.initializer = Key.INITIALIZER["he normal"]()
      elif activation.lower() in normalization:
        self.initializer = Key.INITIALIZER["glorot normal"]()
      else:
        self.initializer = Key.INITIALIZER["default"]()

class Recurrent(Recurrent):
  def __init__(self, cells:int, activation:str, input_sequence:tuple[int,...]=None, output_sequence:tuple[int,...]=None, name:str="Null", *args, **kwargs):
    """
    Recurrent
    -----
      Recurrent layer that processes sequences of data, the input format should be a 2D array (excludes batch dimension), so input_shape should be of the form
      (sequence_length, features). The layer processes the input sequence sequence_length by sequence_length, maintaining a hidden state of size features.
    -----
    Args
    -----
    - Cells                       (int)           : the number of cells in the layer
    - activation                  (string)        : the activation activation for the layer
    - (Optional) input_sequence   (tuple of int)  : indices of cells that receive input from the input sequence, all cells receive input by default
    - (Optional) output_sequence  (tuple of int)  : indices of cells that output to the next layer, all cells output by default
    - (Optional) name             (string)        : the name of the layer
    -----
    activation activations
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - ReTanh

    Normalization activations
    - Softmax
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric activations
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
    
    self.cells = cells
    self.name = name

    if type(activation) == str:
      assert activation in Key.FUNCTION, f"Unknown activation: '{activation}'. Available: {list(Key.FUNCTION.keys())}"
      self.function = Key.FUNCTION[activation.lower()]
    else:
      raise ValueError("activation must be a string or an object with 'forward' and 'backward' methods.")

    self.initializer = Key.INITIALIZER['default']()
    if activation.lower() in rectifiers:
      self.initializer = Key.INITIALIZER['he normal']()
    elif activation.lower() in normalization:
      self.initializer = Key.INITIALIZER['glorot normal']()

    if 'initializer' in kwargs:
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer = Key.INITIALIZER[kwargs['initializer'].lower()]()

    self.input_sequence = input_sequence
    self.output_sequence = output_sequence

class LSTM(LSTM):
  def __init__(self, cells:int, activation:str, input_sequence:tuple[int,...]=None, output_sequence:tuple[int,...]=None, name:str="Null", *args, **kwargs):
    """
    LSTM (Long Short-Term Memory)
    -----
      a Long Short-Term Memory layer that processes sequences of data, the input format should be a 2D array (excludes batch dimension), so input_shape should be of the form
      (sequence_length, features). The layer processes the input sequence sequence_length by sequence_length, maintaining a hidden state of size features.
    -----
    Args
    -----
    - Cells                       (int)           : the number of cells in the layer
    - activation                  (string)        : the activation activation for the layer
    - (Optional) input_sequence   (tuple of int)  : indices of cells that receive input from the input sequence, all cells receive input by default
    - (Optional) output_sequence  (tuple of int)  : indices of cells that output to the next layer, all cells output by default
    - (Optional) name             (string)        : the name of the layer
    -----
    activation activations
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - ReTanh

    Normalization activations
    - Softmax
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric activations
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
    
    self.cells = cells
    self.name = name

    if type(activation) == str:
      assert activation in Key.FUNCTION, f"Unknown activation: '{activation}'. Available: {list(Key.FUNCTION.keys())}"
      self.function = Key.FUNCTION[activation.lower()]
    else:
      raise ValueError("activation must be a string or an object with 'forward' and 'backward' methods.")

    # initializer selection matching your Recurrent
    self.initializer = Key.INITIALIZER['default']()
    if activation.lower() in rectifiers:
      self.initializer = Key.INITIALIZER['he normal']()
    elif activation.lower() in normalization:
      self.initializer = Key.INITIALIZER['glorot normal']()

    if 'initializer' in kwargs:
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer = Key.INITIALIZER[kwargs['initializer'].lower()]()

    self.input_sequence = input_sequence
    self.output_sequence = output_sequence

class GRU(GRU):
  def __init__(self, cells:int, activation:str, input_sequence:tuple[int,...]=None, output_sequence:tuple[int,...]=None, name:str="Null", *args, **kwargs):
    """
    GRU (Gated Recurrent Unit)
    -----
      A Gated Recurrent Unit layer that processes sequences of data, the input format should be a 2D array (excludes batch dimension), so input_shape should be of the form
      (sequence_length, features). The layer processes the input sequence sequence_length by sequence_length, maintaining a hidden state of size features.
    -----
    Args
    -----
    - cells                       (int)           : the number of cells in the layer
    - activation                  (string)        : the activation activation for the layer
    - (Optional) input_sequence   (tuple of int)  : indices of cells that receive input from the input sequence, all cells receive input by default
    - (Optional) output_sequence  (tuple of int)  : indices of cells that output to the next layer, all cells output by default
    - (Optional) name             (string)        : the name of the layer
    -----
    activation activations
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - ReTanh

    Normalization activations
    - Softmax
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
    Parametric activations
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
    self.cells = cells
    self.name = name

    if type(activation) == str:
      assert activation in Key.FUNCTION, f"Unknown activation: '{activation}'. Available: {list(Key.FUNCTION.keys())}"
      self.function = Key.FUNCTION[activation.lower()]
    else:
      raise ValueError("activation must be a string or an object with 'forward' and 'backward' methods.")

    # default init
    self.initializer = Key.INITIALIZER['default']()
    if activation.lower() in rectifiers:
      self.initializer = Key.INITIALIZER['he normal']()
    elif activation.lower() in normalization:
      self.initializer = Key.INITIALIZER['glorot normal']()

    if 'initializer' in kwargs:
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer = Key.INITIALIZER[kwargs['initializer'].lower()]()

    self.input_sequence = input_sequence
    self.output_sequence = output_sequence

class Attention(Attention):
  def __init__(self, heads:int, activation:str, name:str="Null", *args, **kwargs):
    """
    Multiheaded Self-Attention
    -----
      Primary block within Transformer networks, the amount of attention heads is configurable. 
      It accepts data with shape (batch_size, sequence_length, features) simmilar to RNNs.
    -----
    Args
    -----
    - heads           (int)     : the number of attention heads
    - activation      (string)  : the activation activation applied to the output
    - (Optional) name (string)  : the name of the layer
    """
    self.heads = heads
    self.name = name

    if type(activation) == str:
      assert activation in Key.FUNCTION, f"Unknown activation: '{activation}'. Available: {list(Key.FUNCTION.keys())}"
      self.function = Key.FUNCTION[activation.lower()]
    else:
      raise ValueError("activation must be a string or an object with 'forward' and 'backward' methods.")

    # initializer selection
    self.initializer = Key.INITIALIZER['default']()
    if activation.lower() in rectifiers:
      self.initializer = Key.INITIALIZER['he normal']()
    elif activation.lower() in normalization:
      self.initializer = Key.INITIALIZER['glorot normal']()

    if 'initializer' in kwargs:
      if kwargs['initializer'].lower() not in Key.INITIALIZER:
        raise ValueError(f"Unknown initializer: '{kwargs['initializer'].lower()}'. Available: {list(Key.INITIALIZER.keys())}")
      self.initializer = Key.INITIALIZER[kwargs['initializer'].lower()]()

# functional layers

class Operation(Operation):
  def __init__(self, operation:typing.Union[str, callable], name:str="Null", *args, **kwargs):
    """
    Operation
    -----
      A layer that performs an operation on any ndim input while preserving shape. 
      this layer auto-adjusts and does not need to a fixed input shape, but make sure to set the input shape in the format that the operation expects.
    -----
    Args
    -----
    - operation             (str or callable) : the operation to perform on the input
    - operation_derivative  (callable)        : the derivative of the operation if a callable is provided, make sure the backwards activation has the format of activation(error, weighted_sums)
    - (Optional) name       (string)          : the name of the layer
    -----
    activation activations
    - ReLU
    - Softplus
    - Mish
    - Swish
    - Leaky ReLU
    - GELU
    - ReEU
    - None
    - ReTanh

    Normalization activations
    - Softmax
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh

    Scaler activations
    - Standard Scaler
    - Min Max Scaler (frozen to [0,1])
    - Max Abs Scaler
    - Robust Scaler
    """
    
    if type(operation) == str:
      assert operation in Key.FUNCTION, f"Unknown activation: '{operation}'. Available: {list(Key.FUNCTION.keys())}"
      self.function = Key.FUNCTION[operation.lower()]

    else:
      self.function = operation
      
      if not hasattr(operation, "parameter"):
        raise TypeError("operation function must inherit from the Function base class")
    
    self.name = name
