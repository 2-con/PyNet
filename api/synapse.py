"""
Synapse API
=====
  A high-level API for sequential models of neural networks, Synapse automatically handles the learning
  process along with hyperparameters unless specified. things such as input neurons do not need
  to be defined and are handled automatically.
-----
Provides
-----
  (Learnable layers)
  1. Convolution
  2. Dense

  (Utility layers)
  1. Maxpooling
  2. Meanpooling
  3. Flatten
  4. Reshape
  5. AFS (Adaptive Feature Scaler)
  6. Operation (normalization and activation functions)
  
  (Parallelization / Branching)
  7. Parallel
  8. Merge
  
  (Recurrent units)
  9. Recurrent
  10. LSTM
  11. GRU

"""
#######################################################################################################
#                                    File Information and Handling                                    #
#######################################################################################################

__version__ = "1.0.0"

__package__ = "pynet"

if __name__ == "__main__":
  print("""
        This file is not meant to be run as a main file.
        More information can be found about PyNet's Synapse API on the documentation.
        'docs.txt' or https://www.pynet.com/api_docs/python/synapse
        """)
  exit()

#######################################################################################################
#                                               Imports                                               #
#######################################################################################################

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np

from tools import arraytools, scaler, utility, visual
from tools import math as math2

from core import activation as Activation
from core import derivative as Derivative
from core import loss as Error
from core import metric as Metrics
from core import initialization as Initialization
import core.optimizer as optimizer

Optimizer = optimizer.Optimizer # set global object

#######################################################################################################
#                                               Extra                                                 #
#######################################################################################################

""" Notes



"""

""" TODO:

"""

#######################################################################################################
#                                         Internal Functions                                          #
#######################################################################################################

class Key:

  ACTIVATION = {
    # rectifiers
    'relu': Activation.ReLU,
    'softplus': Activation.Softplus,
    'mish': Activation.Mish,
    'swish': Activation.Swish,
    'leaky relu': Activation.Leaky_ReLU,
    'elu': Activation.ELU,
    'gelu': Activation.GELU,
    'selu': Activation.SELU,
    'reeu': Activation.ReEU,
    'none': Activation.Linear,
    'tandip': Activation.TANDIP,

    # normalization functions
    'binary step': Activation.Binary_step,
    'softsign': Activation.Softsign,
    'sigmoid': Activation.Sigmoid,
    'tanh': Activation.Tanh,
  }

  ACTIVATION_DERIVATIVE = {
    # rectifiers
    'relu': Derivative.ReLU_derivative,
    'softplus': Derivative.Softplus_derivative,
    'mish': Derivative.Mish_derivative,
    'swish': Derivative.Swish_derivative,
    'leaky relu': Derivative.Leaky_ReLU_derivative,
    'elu': Derivative.ELU_derivative,
    'gelu': Derivative.GELU_derivative,
    'selu': Derivative.SELU_derivative,
    'reeu': Derivative.ReEU_derivative,
    'none': Derivative.Linear_derivative,
    'tandip': Derivative.TANDIP_derivative,

    # normalization functions
    'binary step': Derivative.Binary_step_derivative,
    'softsign': Derivative.Softsign_derivative,
    'sigmoid': Derivative.Sigmoid_derivative,
    'tanh': Derivative.Tanh_derivative,
  }

  SCALER = {
    'standard scaler': scaler.standard_scaler,
    'min max scaler': scaler.min_max_scaler,
    'max abs scaler': scaler.max_abs_scaler,
    'robust scaler': scaler.robust_scaler,
    'softmax': scaler.softmax,
  }

  SCALER_DERIVATIVE = {
    'standard scaler': Derivative.Standard_scaler_derivative,
    'min max scaler': Derivative.Min_max_scaler_derivative,
    'max abs scaler': Derivative.Max_abs_scaler_derivative,
    'robust scaler': Derivative.Robust_scaler_derivative,
    'softmax': Derivative.Softmax_derivative,
  }

  ERROR = {
    # continuous error
    'mean squared error': Error.Mean_squared_error,
    'mean abseloute error': Error.Mean_absolute_error,
    'total squared error': Error.Total_squared_error,
    'total abseloute error': Error.Total_absolute_error,

    # categorical error
    'categorical crossentropy': Error.Categorical_crossentropy,
    'sparse categorical crossentropy': Error.Sparse_categorical_crossentropy,
    'binary crossentropy': Error.Binary_crossentropy,
    'hinge loss': Error.Hinge_loss,
    'l1 loss': Error.l1_loss,
  }

  OPTIMIZER = {
    'adam': Optimizer.Adam,
    'rmsprop': Optimizer.RMSprop,
    'adagrad': Optimizer.Adagrad,
    'amsgrad': Optimizer.Amsgrad,
    'adadelta': Optimizer.Adadelta,
    'gradclip': Optimizer.Gradclip,
    'adamax': Optimizer.Adamax,
    'sgnd': Optimizer.SGND,
    'default': Optimizer.Default,
    'none': Optimizer.Default,
    'rprop': Optimizer.Rprop,
    'momentum': Optimizer.Momentum,
    'novograd': Optimizer.Novograd,
  }

  METRICS = {

    # classification metrics
    'accuracy': Metrics.Accuracy,
    'precision': Metrics.Precision,
    'recall': Metrics.Recall,
    'f1 score': Metrics.F1_score,
    'roc auc': Metrics.ROC_AUC,
    'r2 score': Metrics.R2_Score,

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
  
#######################################################################################################
#                                          Sequential Model                                           #
#######################################################################################################

class Sequential:

  # pre-processing

  def __init__(self, *args):
    """
    Sequential
    ======
      Sequential model where layers are processed sequentially.
      
      Must contain Synapse layers to be added to the model. either directly through the constructor or through the add() method
    -----

    Available layers:
    - Convolution
    - Maxpooling
    - Meanpooling
    - Flatten
    - Reshape
    - Dense
    - AFS (Adaptive Feature Scaler)
    - Operation
    - Localunit
    
    Recurrent layers:
    - Recurrent
    - LSTM
    - GRU
    
    Parallelization layers:
    - Parallel
    - Merge

    Refer to the documentation for more information.
    """
    self.layers = list(args) if args != None else []

    # defined during compiling
    self.optimizer      = None # name
    self.loss           = None # name
    self.metrics        = None # name
    self.learning_rate  = None
    self.epochs         = None

    self.is_compiled    = False # if the model is already compiled
    self.is_trained     = False # if the model is already fitted
    self.is_validated   = False # if the model is already validated

    self.error_logs = []
    self.validation_error_logs = []

    self.optimizer_instance = Optimizer()

  def add(self, layer):
    """
    Add
    -----
      Adds a specified layer to the network.
    -----
    Args
    -----
    - layer (synapse object) : the layer to add to the model
    """
    if type(layer) not in (Convolution, Dense, Maxpooling, Meanpooling, Flatten, Reshape, AFS, Operation, Localunit):
      raise ValueError("layer must be of type Convolution, Dense, Maxpooling, Meanpooling, Flatten, Reshape, or AFS")

    self.layers.append(layer)

  def compile(self, optimizer, loss, learning_rate, epochs, metrics, **kwargs):
    """
    Compile
    -----
      Compiles the model to be ready for training.
      the PyNet commpiler will automatically take care of hyperparameters and fine tuning under the hood
      unless explicitly defined 
    -----
    Args
    -----
    - optimizer                   (str)   : optimizer to use
    - loss                        (str)   : loss function to use
    - metrics                     (list)  : metrics to use
    - learning_rate               (float) : learning rate to use
    - epochs                      (int)   : number of epochs to train for
    
    - (Optional) early_stopping   (bool)  : whether or not to use early stopping, defaults to False
    - (Optional) patience         (bool)  : how many epochs to wait before early stopping, defaults to 5
    - (Optional) validation       (bool)  : the metrics for validation, defaults to the loss
    - (Optional) validation_split (float) : controls how much of the training data is used for validation, defaults to 0 (ranges from 0 to 1)
    - (Optional) batchsize        (int)   : batch size, defaults to 1
    - (Optional) initialization   (int)   : weight initialization
    - (Optional) experimental     (str)   : experimental settings to use, refer to the documentation for more information
    
    - (optional) verbose          (int) : whether to show anything during training
    - (optional) logging          (int) : how often to show training stats
    
    Optimizer hyperparameters
    -----
    - (Optional) alpha    (float)
    - (Optional) beta     (float)
    - (Optional) epsilon  (float)
    - (Optional) gamma    (float)
    - (Optional) delta    (float)
    -----
    Optimizers
    - ADAM        (Adaptive Moment Estimation)
    - RMSprop     (Root Mean Square Propagation)
    - Adagrad
    - Amsgrad
    - Adadelta
    - Gradclip    (Gradient Clipping)
    - Adamax
    - SGND        (Sign Gradient Descent)
    - Default     (Averaged Gradient)
    - Rprop       (Resilient propagation)
    - Momentum
    - None        (Full Gradient)

    Losses
    - mean squared error
    - mean abseloute error
    - total squared error
    - total abseloute error
    - categorical crossentropy
    - binary cross entropy
    - sparse categorical crossentropy
    - hinge loss

    Metrics
    - accuracy
    - precision
    - recall
    - f1 score
    - roc auc
    - r2 score

    """
    self.optimizer = optimizer.lower()
    self.loss = loss.lower()
    self.metrics = [m.lower() for m in metrics]
    self.learning_rate = learning_rate
    self.epochs = epochs
    
    self.batchsize = kwargs.get('batchsize', 1)
    self.experimental = kwargs.get('experimental', [])
    self.stopping = kwargs.get('early_stopping', False)
    self.patience = kwargs.get('patience', 5)
    self.validation = kwargs.get('validation', loss.lower())
    self.validation_split = kwargs.get('validation_split', 0)
    
    self.verbose = kwargs.get('verbose', 0)
    self.logging = kwargs.get('logging', 1)
    
    self.alpha = kwargs.get('alpha', None) # momentum decay
    self.beta = kwargs.get('beta', None)
    self.epsilon = kwargs.get('epsilon', None) # zerodivison prevention
    self.gamma = kwargs.get('gamma', None)
    self.delta = kwargs.get('delta', None)

    self.is_compiled = True

    # Structure - error prevention
    if self.layers == [] or self.layers is None:
      raise ValueError("No layers in model")
    for layer_index, layer in enumerate(self.layers):
      if type(layer) not in (Dense, Convolution, Maxpooling, Meanpooling, Flatten, Reshape, AFS, Operation, Localunit, Recurrent, LSTM, GRU, RecurrentBlock, Dropout):
        raise TypeError(f"Unknown layer type '{layer.__class__.__name__ }' at layer {layer_index+1}")

    # Kwargs - error prevention
    if learning_rate <= 0:
      raise ValueError("learning_rate must be positive")
    if type(learning_rate) != float:
      raise TypeError("learning_rate must be a float")
    if type(epochs) != int:
      raise TypeError("epochs must be an intiger")
    if epochs <= 0:
      raise ValueError("epochs must be positive")
    if self.batchsize <= 0:
      raise ValueError("batchsize must be greater than 0")
    if self.patience < 1:
      raise ValueError("patience must be greater than or equal to 1")
    if self.validation_split < 0 or self.validation_split > 1:
      raise ValueError("validation split must be between 0 and 1")
    if type(self.stopping) != bool:
      raise ValueError("early stopping must be a boolean")
    if type(self.validation) != str:
      raise ValueError("validation must be a string of a valid metric or loss function")
    if self.logging < 1:
      raise ValueError("logging must be greater than or equal to 1, or set 'verbose' to 0 if the intent is to show nothing")
    if self.verbose < 0:
      raise ValueError("verbose must be greater than or equal to 0, or set 'verbose' to 0 if the intent is to show nothing")
    if self.validation not in Key.METRICS and self.validation not in Key.ERROR:
      raise ValueError("validation must be a string of a valid metric or loss function")
      
    # Compiler - error prevention
    if self.optimizer is None:
      raise ValueError("Optimizer not set")
    if self.optimizer not in Key.OPTIMIZER:
      raise ValueError(f"Invalid optimizer: {self.optimizer}")
    for metric in metrics:
      if (metric not in Key.METRICS) and (metric not in Key.ERROR):
        raise ValueError(f"Invalid metric: {metric}")
    if self.loss not in Key.ERROR:
      raise ValueError(f"Invalid loss: {self.loss}")

  # processing
  
  def fit(self, features, targets):
    """
    Args
    -----
    - features (list)  : the features to use
    - targets  (list)  : the corresponding targets to use
    """
    def Propagate(input):

      x = input[:]
      activations = [x]
      weighted_sums = []

      if self.RNN:
        
        input_index = 0
        output = 0
        outputWS = 0
        for layer in self.layers: # get activations
          
          if layer.accept_input:
            
            output = layer.apply(x[input_index], output)
            
            activations.append([input[input_index], output])
            
            outputWS = layer.get_weighted_sum(x[input_index], output)
            
            input_index += 1
            
          else:
            output = layer.apply(0, output)
            activations.append([0, output])
            outputWS = layer.get_weighted_sum(0, output)
          
          weighted_sums.append(outputWS)
      
      elif self.LSTM:
        
        long_memory = 0
        short_memory = 0
        input_index = 0
        
        for layer in self.layers: # get activations
          
          if layer.accept_input:
            long_memory, short_memory = layer.apply(input[input_index], long_memory, short_memory)
            activations.append([input[input_index], long_memory, short_memory])
            LT, ST, final_long_term, final_short_term, merged_state, gated_long_term, weighted_input, weighted_short_term = layer.get_weighted_sum(input[input_index], long_memory, short_memory)
            input_index += 1
          
          else:
            long_memory, short_memory = layer.apply(0, long_memory, short_memory)
            activations.append([0, long_memory, short_memory])
            LT, ST, final_long_term, final_short_term, merged_state, gated_long_term, weighted_input, weighted_short_term = layer.get_weighted_sum(0, long_memory, short_memory)
          
          weighted_sums.append([LT, ST, final_long_term, final_short_term, merged_state, gated_long_term, weighted_input, weighted_short_term])
      
      elif self.GRU:
        input_index = 0
        output = 0
        outputWS = 0
        for layer in self.layers: # get activations
          
          if layer.accept_input:
            
            output = layer.apply(x[input_index], output)
            
            activations.append([input[input_index], output])
            
            outputWS = layer.get_weighted_sum(x[input_index], output)
            
            input_index += 1
            
          else:
            output = layer.apply(0, output)
            activations.append([0, output])
            outputWS = layer.get_weighted_sum(0, output)
          
          weighted_sums.append(outputWS)
      
      else: # any other type of network
        x = input[:]
        xWS = 0
        for layer in self.layers:
          
          if type(layer) in (Convolution, Dense, AFS, Operation, Localunit):
            xWS = layer.get_weighted_sum(x)
            weighted_sums.append(xWS)
          
          else:
            weighted_sums.append(x)
          
          x = layer.apply(x)
          activations.append(x)
          
      return activations, weighted_sums

    def Backpropagate(activations, weighted_sums, target):

      def index_corrector(index):
        if index < 0:
          return None
        elif index > prev_layer.receptive_field - 1:
          return None
        else:
          return index

      LSTM_incoming_input = []
      
      errors = [0] * ( len(self.layers) )
      initial_gradient = []
      
      if self.RNN or self.GRU:
        # predicted = [cell[1] for cell in activations[1:]]
        
        predicted = []
        for layer_index, lstm_layer in enumerate(self.layers):
          if lstm_layer.return_output:
            predicted.append(activations[layer_index+1][1])
          else:
            predicted.append(0)
            
      elif self.LSTM:
        
        # predicted = [cell[2] for cell in activations[1:]]
        LSTM_incoming_input = [cell[2] for cell in activations[1:]]
        predicted = LSTM_incoming_input
        
        predicted = []
        for layer_index, lstm_layer in enumerate(self.layers):
          if lstm_layer.return_output:
            predicted.append(activations[layer_index+1][2])
          else:
            predicted.append(0)
      
      else:
        predicted = activations[-1]
      
      if True: # calculate the initial gradient

        if self.loss == 'total squared error':
          initial_gradient = [
            (pred - true) 
            for pred, true in zip(predicted, target)
            ]

        elif self.loss == 'mean abseloute error':
          initial_gradient = [
            math2.sgn(pred - true) / len(target) 
            for pred, true in zip(predicted, target)
            ]

        elif self.loss == 'total abseloute error':
          initial_gradient = [
            math2.sgn(pred - true) 
            for pred, true in zip(predicted, target)
            ]

        elif self.loss == 'categorical crossentropy':

          initial_gradient = [
            -true / pred if pred != 0 else -1000 
            for true, pred in zip(target, predicted)
            ]

        elif self.loss == 'sparse categorical crossentropy':

          initial_gradient = [
            -(true == i) / pred if pred != 0 else -1000 
            for i, pred, true in zip(range(len(predicted)), predicted, target)
            ]

        elif self.loss == 'binary crossentropy':

          initial_gradient = [
            -1 / pred if true == 1 and pred != 0 else 1 / (1 - pred) if pred < 1 and pred != 0 else 1000
            for true, pred in zip(target, predicted)
          ]

        elif self.loss == 'hinge loss':
          initial_gradient = [-true if 1-true*pred > 0 else 0 for true, pred in zip(target, predicted)]

        else: # defaults to MSE if loss is ambiguous
          initial_gradient = [(1 * (pred - true)) / len(target) for pred, true in zip(predicted, target)]

        output_layer_errors = []

        if type(self.layers[-1]) == Operation:

          this_WS = weighted_sums[-1]
          this_activations = activations[-1]
          this_layer = self.layers[-1]
          derivative_list = []

          if self.layers[-1].operation == 'min max scaler':
            derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, min=this_layer.minimum, max=this_layer.maximum) for a in this_WS]

            output_layer_errors = [error * derivative for error, derivative in zip(initial_gradient, derivative_list)]

          elif self.layers[-1].operation == 'standard scaler':
            derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, std=this_layer.std) for a in this_WS]

            output_layer_errors = [error * derivative for error, derivative in zip(initial_gradient, derivative_list)]

          elif self.layers[-1].operation == 'max abs scaler':
            derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, max=this_layer.maximum) for a in this_WS]

            output_layer_errors = [error * derivative for error, derivative in zip(initial_gradient, derivative_list)]

          elif self.layers[-1].operation == 'robust scaler':
            derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, q1=this_layer.q1, q3=this_layer.q3) for a in this_WS]

            output_layer_errors = [error * derivative for error, derivative in zip(initial_gradient, derivative_list)]

          elif self.layers[-1].operation == 'softmax':
            derivative_list = Key.SCALER_DERIVATIVE[this_layer.operation](this_activations)

            sqrt_size = int(len(derivative_list)**0.5)
            derivative_list = arraytools.reshape(derivative_list, sqrt_size, sqrt_size)

            num_outputs = len(initial_gradient)
            num_inputs = len(derivative_list[0]) # Assuming derivative_list is already reshaped

            for j in range(num_inputs): # Iterate through the columns of the Jacobian (inputs of softmax)
              result = 0
              for i in range(num_outputs): # Iterate through the rows of the Jacobian (outputs of softmax) and elements of initial_gradient
                result += initial_gradient[i] * derivative_list[i][j]
              output_layer_errors.append(result)

          else: # if its a regular operation

            derivative_list = [Key.ACTIVATION_DERIVATIVE[this_layer.operation](a) for a in this_WS]

            output_layer_errors = [error * derivative for error, derivative in zip(initial_gradient, derivative_list)]

        elif type(self.layers[-1]) in (Dense, AFS, Localunit): # if its not an operation layer

          output_layer_errors = [
              error * Key.ACTIVATION_DERIVATIVE[self.layers[-1].activation](weighted)
              for error, weighted in zip(initial_gradient, weighted_sums[-1])
          ]

        elif type(self.layers[-1]) == Flatten: # if its a flatten layer

          sizex, sizey = self.layers[-1].input_shape

          output_layer_errors = arraytools.reshape(initial_gradient[:], sizex, sizey)

        elif type(self.layers[-1]) == Recurrent: # if its a recurrent layer
          recurrent_output_errors = initial_gradient[:]
          
          error_C = initial_gradient[-1]
          error_D = 0
          total_error = error_C + error_D
          
          derivative = Key.ACTIVATION_DERIVATIVE[self.layers[-1].activation](weighted_sums[-1])
          
          error_Wa = derivative * total_error * activations[-1][0]
          error_Wb = derivative * total_error * activations[-1][1]
          error_B  = derivative * total_error
          error_a  = derivative * total_error * self.layers[-1].carry_weight
          
          output_layer_errors = [error_Wa, error_Wb, error_B, error_a]
        
        elif type(self.layers[-1]) == LSTM: # if its a LSTM layer
          this_layer = self.layers[-1]
          
          L_error   = 0
          S_error   = 0
          out_error = initial_gradient[-1]
          
          incoming_ST, incoming_LT, final_long_term, final_short_term, merged_state, gated_long_term, weighted_input, weighted_short_term = weighted_sums[-1]
          incoming_input = LSTM_incoming_input[-1]
          
          ST_weights = self.layers[-1].short_term_weights
          extra_weights = self.layers[-1].extra_weights
          input_weights = self.layers[-1].input_weights
          
          total_error = out_error + S_error
          
          # calculate OUT
          
          out_we = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term)  * extra_weights[0] * extra_weights[1] * total_error # calculate [we]
          out_a  = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term) * extra_weights[1] * total_error * extra_weights[2]  # calculate [a]
          out_b  = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term) * extra_weights[0] * total_error * extra_weights[2]  # calculate [b]
          
          A = Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term)            * extra_weights[0] * extra_weights[1] * extra_weights[2] * total_error
          B = Key.ACTIVATION["sigmoid"](merged_state[3])            * Key.ACTIVATION_DERIVATIVE["tanh"](final_long_term) * extra_weights[0] * extra_weights[1] * extra_weights[2] * total_error + L_error
          
          # calculate OUT
          
          merged_D = 1 * A 
          short_D = incoming_ST * A
          input_D = incoming_input * A
          
          # calculate INPUT GATE
          
          B_gate_error = B * Key.ACTIVATION["tanh"](merged_state[2])
          C_gate_error = B * Key.ACTIVATION["sigmoid"](merged_state[1])
          
          merged_B = 1 * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) *  B_gate_error
          merged_C = 1 * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2])    * C_gate_error
          
          input_B = incoming_input * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error
          short_B = incoming_ST    * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error
          
          input_C = incoming_input * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * C_gate_error
          short_C = incoming_ST    * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * C_gate_error
          
          # calculate FORGET GATE
          
          merged_A = 1              * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B
          short_A  = incoming_ST    * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B
          input_A  = incoming_input * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B
          
          # calculate CARRY
          
          carry_ST = (ST_weights[0] * A) + \
                     (ST_weights[1] * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error) + \
                     (ST_weights[2] * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2])    * C_gate_error) + \
                     (ST_weights[3] * A)
          
          carry_LT = B * Key.ACTIVATION["sigmoid"](merged_state[0])
          
          output_layer_errors = [carry_LT, carry_ST, 
                                 [merged_A, merged_B, merged_C, merged_D], 
                                 [short_A, short_B, short_C, short_D], 
                                 [input_A, input_B, input_C, input_D], 
                                 [out_a, out_b, out_we]
                                ]
          
        elif type(self.layers[-1]) == GRU: # if its a GRU layer
          this_layer = self.layers[-1]
          carry_error = 0
          out_error = initial_gradient[-1]
          
          input_weights = this_layer.input_weights
          carry_weights = this_layer.carry_weights
          
          total_error = out_error + carry_error
          final_output, gated_carry, merged_state, weighted_input, weighted_carry, incoming_input, incoming_carry = weighted_sums[-1]
          
          # output gate
          
          error_C = Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * ( 1 - Key.ACTIVATION["sigmoid"](merged_state[1]) ) * total_error
          
          bias_C  = 1                                                             * error_C
          input_C = incoming_input                                                * error_C
          carry_C = (Key.ACTIVATION["sigmoid"](merged_state[0]) * incoming_carry) * error_C
          
          # update gate
          
          error_B = total_error * (incoming_carry - Key.ACTIVATION["tanh"](merged_state[2])) * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1])
                    
          bias_B  = 1              * error_B
          input_B = incoming_input * error_B
          carry_B = incoming_carry * error_B
          
          # reset gate
          
          error_A = total_error * (1 - Key.ACTIVATION["sigmoid"](merged_state[1])) * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * incoming_carry * carry_weights[2] * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0])

          bias_A  = 1              * error_A
          input_A = incoming_input * error_A
          carry_A = incoming_carry * error_A
          
          # calculate upstream gradient
          
          carry_error = ( carry_weights[0] * error_A                                                                                                                                                          ) + \
                        ( carry_weights[1] * error_B                                                                                                                                                          ) + \
                        ( total_error * (1 - Key.ACTIVATION["sigmoid"](merged_state[1])) * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * Key.ACTIVATION["sigmoid"](merged_state[0]) * carry_weights[2] ) + \
                        ( total_error * Key.ACTIVATION["sigmoid"](merged_state[1])                                                                                                                            ) 
          
          output_layer_errors = [
                                  carry_error, 
                                 [bias_A, bias_B, bias_C], 
                                 [carry_A, carry_B, carry_C], 
                                 [input_A, input_B, input_C]
                                ]
        
        elif type(self.layers[-1]) == RecurrentBlock:
          raise NotImplementedError("WIP")
          # work in progress #########################################
          
        else:
          raise NotImplementedError(f"Layer {self.layers[-1].__class__.__name__} as the last layer is not supported.")
        
      errors[-1] = output_layer_errors

      for this_layer_index in reversed(range(len(self.layers)-1)):
        # - FRONT | next layer --> this layer --> previous layer | BACK +

        this_layer = self.layers[this_layer_index]
        next_layer = self.layers[this_layer_index - 1] if this_layer_index > 0 else None
        
        this_activations = activations[this_layer_index]
        this_WS = weighted_sums[this_layer_index]

        prev_errors = errors[this_layer_index + 1]
        prev_layer = self.layers[this_layer_index + 1]

        layer_errors = []
        
        ######################################################################################
        if type(this_layer) in (Dense, AFS, Localunit):
          
          if type(prev_layer) == Dense:
            
            layer_errors = [
            Key.ACTIVATION_DERIVATIVE[this_layer.activation](this_WS[this_neuron_index]) *
            sum(
                prev_errors[prev_neuron_index] * prev_neuron['weights'][this_neuron_index]
                for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons)
            )
            for this_neuron_index, _ in enumerate(this_layer.neurons)
              ]

          elif type(prev_layer) == AFS:
            layer_errors = [
            Key.ACTIVATION_DERIVATIVE[this_layer.activation](this_WS[this_neuron_index]) *
            prev_errors[this_neuron_index] * prev_layer.neurons[this_neuron_index]['weight']
            for this_neuron_index, _ in enumerate(this_layer.neurons)
            ]
            
          elif type(prev_layer) == Operation:
            layer_errors = [
                  Key.ACTIVATION_DERIVATIVE[this_layer.activation](this_WS[this_neuron_index]) *
                  prev_errors[this_neuron_index]
                  for this_neuron_index, _ in enumerate(this_layer.neurons)
              ]

          elif type(prev_layer) == Localunit:
            
            layer_errors = [
                Key.ACTIVATION_DERIVATIVE[this_layer.activation](this_WS[this_neuron_index]) *
                sum(
                    prev_errors[prev_neuron_index] * prev_neuron['weights'][
                        index_corrector(this_neuron_index - prev_neuron_index, prev_layer.receptive_field) # Pass receptive_field
                    ]
                    for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons)
                    if index_corrector(this_neuron_index - prev_neuron_index, prev_layer.receptive_field) is not None # Check condition here
                )
                for this_neuron_index, _ in enumerate(this_layer.neurons)
            ]
            
          elif type(prev_layer) == Dropout:
            layer_errors = [ error * Key.ACTIVATION_DERIVATIVE[this_layer.activation](WS) for error, WS in zip(prev_errors, this_WS) ]
          
          else:
            raise TypeError(f"Layer {this_layer_index+1} cannot perform backpropagation on a {prev_layer.__name__} (layer {this_layer_index+2})")
          
        ######################################################################################
        elif type(this_layer) == Dropout:
          
          neuron_error = []
          
          if type(prev_layer) in (Dense, AFS, Operation, Localunit):
            
            if type(prev_layer) == Dense:
              
              for neuron_index, _ in enumerate(this_layer.mask):
                for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):
                  neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][neuron_index])

                layer_errors.append( sum(neuron_error) )
            
            elif type(prev_layer) == AFS:
              for neuron_index, _ in enumerate(this_layer.mask):
                for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):
                  neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weight'])

                layer_errors.append( sum(neuron_error) )

            elif type(prev_layer) == Operation:
              for neuron_index, _ in enumerate(this_layer.mask):
                for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):
                  neuron_error.append(prev_errors[prev_neuron_index])

                layer_errors.append( sum(neuron_error) )

            elif type(prev_layer) == Localunit:
              for neuron_index, _ in enumerate(this_layer.mask):
                for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

                  if index_corrector(this_neuron_index-(prev_neuron_index)) != None:
                    neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][neuron_index - prev_neuron_index])

                layer_errors.append( sum(neuron_error) )

            # process the erors through the dropout mask
            layer_errors = [ error * mask for error, mask in zip(layer_errors, this_layer.mask)]
            
          else: # probably a 2D image processing layer
            layer_errors = prev_errors
            
            # process the erors through the dropout mask
            layer_errors = [
              [ error * mask for error, mask in zip(row, row_mask)] 
              for row, row_mask in zip(layer_errors, this_layer.mask)
            ]
          
        ###################################################################################### TO BE OVERHAULED
        elif type(this_layer) == Convolution:
          
          updater_total_err = []
          updater_err = []
          
          raw_errs = []
          
          for channel, previous_err in zip(this_layer.kernels, prev_errors):
            
            updater_err = []
            
            for kernel in channel:
              
              if 'BPP_kernel_flip' in self.experimental:
                kernel = arraytools.mirror(this_layer.kernel[:], 'X')
              
              smol_err = arraytools.generate_array(this_layer.input_shape[0], this_layer.input_shape[1], value=0)
              
              for a in range((this_layer.input_shape[0]) - (len(kernel) - 1)):
                for b in range((this_layer.input_shape[1]) - (len(kernel[0]) - 1)):
                  
                  # Apply the flipped kernel to the corresponding region in layer_errors
                  for kernel_row in range(len(kernel)):
                    for kernel_col in range(len(kernel[0])):
                      
                      # Calculate the corresponding position in layer_errors
                      output_row = a + kernel_row
                      output_col = b + kernel_col
                      
                      # Accumulate the weighted error
                      smol_err[output_row][output_col] += previous_err[a][b] * kernel[kernel_row][kernel_col] * Key.ACTIVATION_DERIVATIVE[this_layer.activation](this_WS[a][b])

              updater_err.append(smol_err)
              raw_errs.append(smol_err)
            
            updater_total_err.append(updater_err)
            
          # forward pass gradient section
          for a in range(len(raw_errs)//this_layer.channels):
            totaled = arraytools.generate_array(this_layer.input_shape[0], this_layer.input_shape[1], value=0)
            
            for b in range(a, len(raw_errs), this_layer.input_shape[2]):
              totaled = arraytools.total(totaled, raw_errs[b])

            layer_errors.append(totaled)
            
          # if 'BPP_kernel_flip' in self.experimental:
          #   kernel = arraytools.mirror(this_layer.kernel[:], 'X')
          # else:
          #   kernel = this_layer.kernel[:]

          # layer_errors = arraytools.generate_array(value=0, *this_layer.input_shape)

          # for a in range(len(layer_errors) - (len(kernel) - 1)):
          #   for b in range(len(layer_errors[0]) - (len(kernel[0]) - 1)):

          #     # Apply the flipped kernel to the corresponding region in layer_errors
          #     for kernel_row in range(len(kernel)):
          #       for kernel_col in range(len(kernel[0])):
                  
          #         # Calculate the corresponding position in layer_errors
          #         output_row = a + kernel_row
          #         output_col = b + kernel_col
                  
          #         # Accumulate the weighted error
          #         layer_errors[output_row][output_col] += prev_errors[a][b] * kernel[kernel_row][kernel_col] * Key.ACTIVATION_DERIVATIVE[this_layer.activation](this_WS[a][b])
        
        ######################################################################################
        elif type(this_layer) == Flatten:
          
          if type(prev_layer) in (Dense, AFS, Operation, Localunit):
            for this_neuron_index in range(len(this_layer.neurons)):

              neuron_error = []

              if type(prev_layer) == Dense:
                
                for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

                  neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][this_neuron_index])

              elif type(prev_layer) == AFS:
                for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

                  neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weight'])

              elif type(prev_layer) == Operation:
                for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

                  neuron_error.append(prev_errors[prev_neuron_index])

              elif type(prev_layer) == Localunit:
                for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

                  if index_corrector(this_neuron_index-(prev_neuron_index)) != None:
                    neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][this_neuron_index-prev_neuron_index])

              layer_errors.append( sum(neuron_error) )

          else:
            layer_errors = prev_errors
          
          layer_errors = arraytools.reshape(layer_errors[:], this_layer.input_shape)
          
          # force into a 2D array
          if len(arraytools.shape(layer_errors)) == 1:
            layer_errors = [layer_errors]
          
        ######################################################################################
        elif type(this_layer) == Reshape:
          
          layer_errors = arraytools.reshape(prev_errors[:], this_layer.input_shape)

        ###################################################################################### REWORK
        elif type(this_layer) == Maxpooling:
          thisx, thisy = this_layer.input_size

          layer_errors = arraytools.generate_array(thisx, thisy, value=0)

          skibidi = -1

          # iterate over all the layers
          for a in range(0, len(weighted_sums[this_layer_index-1]), this_layer.stride):

            skibidi += 1
            toilet = -1

            # # iterate over all the elements in the layer
            for b in range(0, len(weighted_sums[this_layer_index-1][a]), this_layer.stride):
              toilet += 1

              max_value = this_WS[skibidi][toilet]  # Get the maximum value
              count = 0  # Count of elements with the maximum value

              # Find all elements with the maximum value in the pooling window
              for c in range(this_layer.size):
                for d in range(this_layer.size):
                  if a + c < len(weighted_sums[this_layer_index - 1]) and b + d < len(weighted_sums[this_layer_index - 1][a]):
                    if weighted_sums[this_layer_index - 1][a + c][b + d] == max_value:
                      count += 1

              # Distribute the gradient equally among the maximum elements
              for c in range(this_layer.size):
                for d in range(this_layer.size):
                  if a + c < len(weighted_sums[this_layer_index - 1]) and b + d < len(weighted_sums[this_layer_index - 1][a]):
                    if weighted_sums[this_layer_index - 1][a + c][b + d] == max_value:
                      layer_errors[a + c][b + d] += prev_errors[skibidi][toilet] / count  # Divide by count

        ###################################################################################### REWORK
        elif type(this_layer) == Meanpooling:

          thisx, thisy = this_layer.input_size

          layer_errors = arraytools.generate_array(thisx, thisy, value=0)

          # for meanpooling, we need to distribute the error over the kernel size
          # this is done by dividing the error by the kernel size (averaging it over the kernel size)

          skibidi = -1

          # iterate over all the layers
          for a in range(0, len(weighted_sums[this_layer_index-1]), this_layer.stride):

            skibidi += 1
            toilet = -1

            # iterate over all the elements in the layer
            for b in range(0, len(weighted_sums[this_layer_index-1][a]), this_layer.stride):
              toilet += 1

              # control the vertical
              for c in range(this_layer.size):

                # control the horizontal
                for d in range(this_layer.size):

                  if a+c < len(weighted_sums[this_layer_index-1]) and b+d < len(weighted_sums[this_layer_index-1][a]):

                    # distribute the error over the kernel size

                    layer_errors[a+c][b+d] += prev_errors[skibidi][toilet] / (this_layer.size**2)

        ######################################################################################
        elif type(this_layer) == Operation:

          if this_layer.operation == 'min max scaler':
            derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, min=this_layer.minimum, max=this_layer.maximum) for a in this_WS]

          elif this_layer.operation == 'standard scaler':
            derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, std=this_layer.std) for a in this_WS]

          elif this_layer.operation == 'max abs scaler':
            derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, max=this_layer.maximum) for a in this_WS]

          elif this_layer.operation == 'robust scaler':
            derivative_list = [Key.SCALER_DERIVATIVE[this_layer.operation](a, q1=this_layer.q1, q3=this_layer.q3) for a in this_WS]

          elif this_layer.operation == 'softmax':
            derivative_list = Key.SCALER_DERIVATIVE[this_layer.operation](this_activations)

          for this_neuron_index in range(len(this_layer.neurons)):

            neuron_error = []

            if this_layer.operation in Key.ACTIVATION_DERIVATIVE:
              derivative = Key.ACTIVATION_DERIVATIVE[this_layer.operation](this_WS[this_neuron_index])
            else:
              derivative = derivative_list[this_neuron_index]

            if type(prev_layer) == Dense:

              for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):
                neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][this_neuron_index])

              layer_errors.append( derivative * sum(neuron_error) )

            elif type(prev_layer) == AFS:

              layer_errors.append( derivative * prev_errors[this_neuron_index] * prev_layer.neurons[this_neuron_index]['weight'] )

            elif type(prev_layer) == Operation:
              layer_errors.append( derivative * prev_errors[this_neuron_index] )

            elif type(prev_layer) == Localunit:

              for prev_neuron_index, prev_neuron in enumerate(prev_layer.neurons):

                if index_corrector(this_neuron_index-(prev_neuron_index)) != None:
                  neuron_error.append(prev_errors[prev_neuron_index] * prev_neuron['weights'][this_neuron_index-prev_neuron_index])

              layer_errors.append( derivative * sum(neuron_error) )

            elif type(prev_layer) == Reshape:

              layer_errors.append( prev_errors[this_neuron_index] )

        # recurrent layers
        # always in isolation unless blocked
        ######################################################################################
        elif type(this_layer) == Recurrent:
          if type(prev_layer) != Recurrent:
            raise SystemError(f'Recurrent layer must be preceded by a Recurrent layer and not {type(prev_layer)}')
          
          error_C = recurrent_output_errors[this_layer_index]
          error_D = prev_errors[3]
          
          if this_layer.return_output:
            total_error = error_C + error_D
          else:
            total_error = error_D
          
          derivative = Key.ACTIVATION_DERIVATIVE[this_layer.activation](this_WS)
          
          error_Wa = derivative * total_error * this_activations[0]
          error_Wb = derivative * total_error * this_activations[1]
          error_B  = derivative * total_error
          error_a  = derivative * total_error * this_layer.carry_weight
          
          layer_errors = [error_Wa, error_Wb, error_B, error_a]
        
        ######################################################################################
        elif type(this_layer) == LSTM:
          if type(prev_layer) != LSTM:
            raise SystemError(f'LSTM layer must be preceded by a LSTM layer and not {type(prev_layer)}')
          
          L_error   = prev_errors[0]
          S_error   = prev_errors[1]
          out_error = initial_gradient[this_layer_index] if this_layer.return_output else 0
          
          incoming_ST, incoming_LT, final_long_term, final_short_term, merged_state, gated_long_term, weighted_input, weighted_short_term = weighted_sums[this_layer_index]
          incoming_input = LSTM_incoming_input[this_layer_index]
          
          ST_weights    = this_layer.short_term_weights
          extra_weights = this_layer.extra_weights
          input_weights = this_layer.input_weights
          
          total_error = out_error + S_error
          
          # calculate OUT
          
          out_we = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term)  * extra_weights[0] * extra_weights[1] * total_error # calculate [we]
          out_a  = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term) * extra_weights[1] * total_error * extra_weights[2]  # calculate [a]
          out_b  = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term) * extra_weights[0] * total_error * extra_weights[2]  # calculate [b]
          
          A = Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term)            * extra_weights[0] * extra_weights[1] * extra_weights[2] * total_error
          B = Key.ACTIVATION["sigmoid"](merged_state[3])            * Key.ACTIVATION_DERIVATIVE["tanh"](final_long_term) * extra_weights[0] * extra_weights[1] * extra_weights[2] * total_error + L_error
          
          # calculate OUT
          
          merged_D = 1 * A 
          short_D = incoming_ST * A
          input_D = incoming_input * A
          
          # calculate INPUT GATE
          
          B_gate_error = B * Key.ACTIVATION["tanh"](merged_state[2])
          C_gate_error = B * Key.ACTIVATION["sigmoid"](merged_state[1])
          
          merged_B = 1 * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error
          merged_C = 1 * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2])    * C_gate_error
          
          input_B = incoming_input * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error
          short_B = incoming_ST    * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error
          
          input_C = incoming_input * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * C_gate_error
          short_C = incoming_ST    * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * C_gate_error
          
          # calculate FORGET GATE
          
          merged_A = 1              * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B 
          short_A  = incoming_ST    * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B
          input_A  = incoming_input * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B
          
          # calculate CARRY
          
          carry_ST = (ST_weights[0] * A) + \
                     (ST_weights[1] * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error) + \
                     (ST_weights[2] * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2])    * C_gate_error) + \
                     (ST_weights[3] * A)
          
          carry_LT = B * Key.ACTIVATION["sigmoid"](merged_state[0])
          
          layer_errors = [carry_LT, carry_ST, 
                          [merged_A, merged_B, merged_C, merged_D], 
                          [short_A, short_B, short_C, short_D], 
                          [input_A, input_B, input_C, input_D], 
                          [out_a, out_b, out_we]
                         ]
        
        ######################################################################################
        elif type(this_layer) == GRU:
          if type(prev_layer) != GRU:
            raise SystemError(f'GRU layer must be preceded by a GRU layer and not {type(prev_layer)}')

          carry_error = prev_errors[0]
          out_error   = initial_gradient[this_layer_index]
          
          input_weights = this_layer.input_weights
          carry_weights = this_layer.carry_weights
          
          total_error = out_error + carry_error
          final_output, gated_carry, merged_state, weighted_input, weighted_carry, incoming_input, incoming_carry = weighted_sums[this_layer_index]
          
          # output gate
          
          error_C = Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * ( 1 - Key.ACTIVATION["sigmoid"](merged_state[1]) ) * total_error
          
          bias_C  = 1                                                             * error_C
          input_C = incoming_input                                                * error_C
          carry_C = (Key.ACTIVATION["sigmoid"](merged_state[0]) * incoming_carry) * error_C
          
          # update gate
          
          error_B = total_error * (incoming_carry - Key.ACTIVATION["tanh"](merged_state[2])) * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1])
                    
          bias_B  = 1              * error_B
          input_B = incoming_input * error_B
          carry_B = incoming_carry * error_B
          
          # reset gate
          
          error_A = total_error * (1 - Key.ACTIVATION["sigmoid"](merged_state[1])) * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * incoming_carry * carry_weights[2] * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0])

          bias_A  = 1              * error_A
          input_A = incoming_input * error_A
          carry_A = incoming_carry * error_A
          
          # calculate upstream gradient
          
          carry_error = ( carry_weights[0] * error_A                                                                                                                                                          ) + \
                        ( carry_weights[1] * error_B                                                                                                                                                          ) + \
                        ( total_error * (1 - Key.ACTIVATION["sigmoid"](merged_state[1])) * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * Key.ACTIVATION["sigmoid"](merged_state[0]) * carry_weights[2] ) + \
                        ( total_error * Key.ACTIVATION["sigmoid"](merged_state[1])                                                                                                                            ) 
          
          layer_errors = [
                          carry_error, 
                          [bias_A, bias_B, bias_C], 
                          [carry_A, carry_B, carry_C], 
                          [input_A, input_B, input_C]
                         ]
        
        ######################################################################################
        elif type(this_layer) == RecurrentBlock:
            
          layer_errors = this_layer.internal(2, prev_errors, LSTM_incoming_input=LSTM_incoming_input)
             
        errors[this_layer_index] = layer_errors[:]

      return errors
    
    def update(activations, weighted_sum, errors, timestep):

      alpha = self.alpha
      beta = self.beta
      epsilon = self.epsilon
      gamma = self.gamma
      delta = self.delta

      optimize = Key.OPTIMIZER.get(self.optimizer)
      learning_rate = self.learning_rate
      param_id = 0 # must be a positive integer
      batchsize = self.batchsize
      
      for this_layer_index in reversed(range(len(self.layers))):

        layer = self.layers[this_layer_index]
        prev_activations = activations[this_layer_index][:]
        this_WS = weighted_sum[this_layer_index]

        if type(layer) == Dense and layer.learnable:

          for neuron_index, neuron in enumerate(layer.neurons):

            for weight_index, weight in enumerate(neuron['weights']):

              # calculate universal gradient
              weight_gradient = 0
              
              for error_versions in errors:
                
                weight_gradient += error_versions[this_layer_index][neuron_index] * prev_activations[weight_index]

              if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
                weight_gradient /= batchsize

              # Update weights
              param_id += 1
              neuron['weights'][weight_index] = optimize(learning_rate, weight, weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

            # Updating bias
            bias_gradient = 0

            for error_versions in errors:
              bias_gradient += error_versions[this_layer_index][neuron_index]

            if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
              bias_gradient /= batchsize

            param_id += 1
            neuron['bias'] = optimize(learning_rate, neuron['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

        elif type(layer) == Convolution and layer.learnable:
          
          bias_gradients = arraytools.generate_array(*arraytools.shape(layer.bias), value=0)
          
          overall_errors = []
          
          for error_versions in errors:

            for channel_index, channel in enumerate(layer.kernels):
              
              channel_errors = []
              
              if len(arraytools.shape(prev_activations)) == 2:
                prev_activations = [prev_activations for _ in range(layer.channels)]
              
              for kernel_index, (raw_kernel, error, activation) in enumerate(zip(channel, error_versions[this_layer_index], prev_activations)):

                weight_gradient = arraytools.generate_array(len(raw_kernel[0]), len(raw_kernel), value=0)
                kernel = arraytools.mirror(raw_kernel[::-1], 'Y') # flipped kernel

                # Rows of prev_errors (0 to 1)
                for a in range((len(activation) - len(kernel)) + 1):

                  # Columns of prev_errors (0 to 1)
                  for b in range((len(activation[0]) - len(kernel[0])) + 1):

                    # Apply the flipped kernel to the corresponding region in layer_errors
                    for kernel_row in range(len(kernel)):

                      for kernel_col in range(len(kernel[0])):
                        # Calculate the corresponding position in layer_errors

                        conv_row = kernel_row + a
                        conv_col = kernel_col + b

                        derivative = Key.ACTIVATION_DERIVATIVE[layer.activation](this_WS[a][b])
                        
                        weight_gradient[kernel_row][kernel_col] += activation[conv_row][conv_col] * error[a][b] * derivative

                # find bias error
                for row in range(len(error_versions[this_layer_index])):
                  for col in range(len(error_versions[this_layer_index][0])):
                    bias_gradients[channel_index][kernel_index] += error[row][col]

                bias_gradients[channel_index][kernel_index] /= arraytools.shape(activation)[0] * arraytools.shape(activation)[1]
                channel_errors.append(weight_gradient)
              
              overall_errors.append(channel_errors)

          # update kernel weights
          for channel_index,  (channel, channel_error, bias_sublist) in enumerate(zip(layer.kernels, overall_errors, bias_gradients)):
            for kernel_index, (kernel, kernel_error, bias_err) in enumerate(zip(channel, channel_error, bias_sublist)):
              
              for a in range(len(kernel)):
                for b in range(len(kernel[a])):

                  param_id += 1

                  if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
                    weight_gradient[a][b] /= batchsize

                  kernel[a][b] = optimize(learning_rate, kernel[a][b], kernel_error[a][b], self.optimizer_instance, 
                                          alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, 
                                          param_id=param_id, timestep=timestep)

              if layer.use_bias:
                param_id += 1

                if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
                  bias_gradient /= batchsize

                layer.bias[channel_index][kernel_index] = optimize(learning_rate, layer.bias[channel_index][kernel_index], bias_err, self.optimizer_instance, 
                                                                   alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, 
                                                                   param_id=param_id, timestep=timestep)

        elif type(layer) == AFS and layer.learnable:

          for this_neuron_index, _ in enumerate(layer.neurons):

            # calculate weight gradient

            weight_gradient = 0

            for error_versions in errors:
              weight_gradient += error_versions[this_layer_index][this_neuron_index] * prev_activations[this_neuron_index]

            if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
              weight_gradient /= batchsize

            param_id += 1
            layer.neurons[this_neuron_index]['weight'] = optimize(learning_rate, layer.neurons[this_neuron_index]['weight'], weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

            # calculate bias gradient
            if layer.use_bias:
              bias_gradient = 0

              for error_versions in errors:
                bias_gradient += error_versions[this_layer_index][this_neuron_index]

              if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
                bias_gradient /= batchsize

              param_id += 1
              layer.neurons[this_neuron_index]['bias'] = optimize(learning_rate, layer.neurons[this_neuron_index]['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

        elif type(layer) == Localunit and layer.learnable:

          for neuron_index, neuron in enumerate(layer.neurons):

            for weight_index in range(len(neuron['weights'])):

              # calculate universal gradient
              weight_gradient = 0

              for error_versions in errors:
                weight_gradient += error_versions[this_layer_index][neuron_index] * prev_activations[weight_index]

              if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
                weight_gradient /= batchsize

              # Update weights
              param_id += 1
              neuron['weights'][weight_index] = optimize(learning_rate, neuron['weights'][weight_index], weight_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

            # Updating bias

            bias_gradient = 0

            for error_versions in errors:
              bias_gradient += error_versions[this_layer_index][neuron_index]

            if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
              bias_gradient /= batchsize

            param_id += 1
            neuron['bias'] = optimize(learning_rate, neuron['bias'], bias_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

        elif type(layer) == Recurrent and layer.learnable:
          Wa_gradient = 0
          Wb_gradient = 0
          B_gradient  = 0
          
          for error_version in errors:
            Wa_gradient += error_version[this_layer_index][0]
            Wb_gradient += error_version[this_layer_index][1]
            B_gradient  += error_version[this_layer_index][2]

          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            Wa_gradient /= batchsize
            Wb_gradient /= batchsize
            B_gradient  /= batchsize

          param_id += 1
          layer.carry_weight = optimize(learning_rate, layer.carry_weight, Wa_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
          param_id += 1
          layer.input_weight = optimize(learning_rate, layer.input_weight, Wb_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
          param_id += 1
          layer.bias         = optimize(learning_rate, layer.bias, B_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
        
        elif type(layer) == LSTM and layer.learnable:
          
          B_ERR     = [0] * 4 # biases (merge)
          ST_ERR    = [0] * 4 # short term weights
          INPUT_ERR = [0] * 4 # input weights
          EXTRA_ERR = [0] * 3 # extra weights (output gate)
          
          for error_version in errors:
            B_ERR     = [B_ERR[i]     + error_version[this_layer_index][2][i] for i in range(4)]
            ST_ERR    = [ST_ERR[i]    + error_version[this_layer_index][3][i] for i in range(4)]
            INPUT_ERR = [INPUT_ERR[i] + error_version[this_layer_index][4][i] for i in range(4)]
            EXTRA_ERR = [EXTRA_ERR[i] + error_version[this_layer_index][5][i] for i in range(3)]
            
          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            B_ERR     = [a / batchsize for a in B_ERR]
            ST_ERR    = [b / batchsize for b in ST_ERR]
            INPUT_ERR = [c / batchsize for c in INPUT_ERR]
            EXTRA_ERR = [d / batchsize for d in EXTRA_ERR]
          
          for index, bias in enumerate(layer.biases):
            param_id += 1
            layer.biases[index] = optimize(learning_rate, bias, B_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
          
          for index, weight in enumerate(layer.short_term_weights):
            param_id += 1
            layer.short_term_weights[index] = optimize(learning_rate, weight, ST_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
          
          for index, weight in enumerate(layer.input_weights):
            param_id += 1
            layer.input_weights[index] = optimize(learning_rate, weight, INPUT_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
          
          for index, weight in enumerate(layer.extra_weights):
            param_id += 1
            if layer.version == 'statquest':
              layer.extra_weights[index] = optimize(learning_rate, weight, EXTRA_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
        
        elif type(layer) == GRU and layer.learnable:
          
          B_ERR     = [0] * 3 # biases
          C_ERR     = [0] * 3 # short term weights
          INPUT_ERR = [0] * 3 # input weights
          
          for error_version in errors:
            B_ERR     = [B_ERR[i]     + error_version[this_layer_index][1][i] for i in range(3)]
            C_ERR     = [C_ERR[i]     + error_version[this_layer_index][2][i] for i in range(3)]
            INPUT_ERR = [INPUT_ERR[i] + error_version[this_layer_index][3][i] for i in range(3)]
            
          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            B_ERR     = [a / batchsize for a in B_ERR]
            C_ERR     = [b / batchsize for b in C_ERR]
            INPUT_ERR = [c / batchsize for c in INPUT_ERR]
          
          for index, bias in enumerate(layer.biases):
            param_id += 1
            layer.biases[index] = optimize(learning_rate, bias, B_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

          for index, weight in enumerate(layer.carry_weights):
            param_id += 1
            layer.carry_weights[index] = optimize(learning_rate, weight, C_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

          for index, weight in enumerate(layer.input_weights):
            param_id += 1
            layer.input_weights[index] = optimize(learning_rate, weight, INPUT_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
        
        elif type(layer) == RecurrentBlock:
          
          layer.internal(3, learning_rate, timestep, self.batchsize)
      
    #################################################################################################
    #                                     Automatic Fitting                                         #
    #################################################################################################
    
    self.is_trained = True
    epochs = self.epochs + 1
    
    train_amount = int(self.validation_split * len(features))
    validation_features = features[:train_amount][:]
    validation_targets  = targets[:train_amount][:]
    
    features  = features[train_amount:][:]
    targets   = targets[train_amount:][:]
    
    if True: # Error Prevention

      # Function args - error prevention
      if not type(features) in (list, tuple):
        raise TypeError("features must be a list")
      if not type(targets) in (list, tuple):
        raise TypeError("targets must be a list")
      if len(features) == 0 or len(targets) == 0:
        raise ValueError("features or targets must not be empty")
      if any(x == [] for x in features) or any(x == [] for x in targets):
        raise ValueError("feature or target must not be empty")

    # neural network fitting
    x = features[0]
    sizes = [arraytools.shape(x)]
    calibrate = False
    errors = []
    timestep = 1
    
    # calibration and structure intialization
    for _ in range(2):
      
      x = features[0]
      
      for layer_index, layer in enumerate(self.layers):

        if type(layer) in (Maxpooling, Meanpooling, Reshape, Dropout):
          
          x = layer.apply(x)
          sizes.append(arraytools.shape(x))
        
        elif type(layer) == Convolution:
          
          if layer_index == 0: # first layer
            layer.reshape_input_shape( list( arraytools.shape(x) ) + [1] )

          else: # every other layer
            layer.reshape_input_shape(arraytools.shape(x))
            
          x = layer.apply(x)
          
          sizes.append(arraytools.shape(x))
        
        elif type(layer) == Flatten:
          x = layer.apply(x)
          layer.set_length(len(arraytools.flatten(x)))
          sizes.append(arraytools.shape(x))

        elif type(layer) in (AFS, Dense, Operation, Localunit):
          
          if calibrate:
            
            x = layer.apply(x)
            
            if layer_index == len(self.layers) - 1: # last
              layer.reshape_input_shape(sizes[layer_index][0], sizes[layer_index])
            
            elif layer_index == 0: # first layer
              layer.reshape_input_shape(len(features[0]), sizes[layer_index+1])

            else: # every other layer
              layer.reshape_input_shape(sizes[layer_index][0], sizes[layer_index+1])

          else:
                      
              if layer_index == 0: # first layer
                layer.reshape_input_shape(len(features[0]), (1,1))

              else: # every other layer
                layer.reshape_input_shape(sizes[layer_index][0], (1,1))
              
              x = layer.apply(x)
              sizes.append(arraytools.shape(x))

        elif type(layer) == RecurrentBlock:
          
          layer.calibrate(self.optimizer, self.loss, self.learning_rate)
          x = layer.apply(x)
          sizes.append(arraytools.shape(x))
            
      calibrate = True
      
      self.sizes = sizes
    
    self.RNN  = any(type(layer) == Recurrent for layer in self.layers)
    self.LSTM = any(type(layer) == LSTM for layer in self.layers)
    self.GRU  = any(type(layer) == GRU for layer in self.layers)
    
    # main training loop - iterate over the epochs
    for epoch in utility.progress_bar(range(epochs), "> Training", "Complete", decimals=2, length=70, empty=' ') if self.verbose==1 else range(epochs):
      epoch_loss = 0
      
      # main training section - iterate over the entire dataset
      for base_index in utility.progress_bar(range(0, len(features), self.batchsize), "> Processing Batch", f"Epoch {epoch+1 if epoch == 0 else epoch}/{epochs-1} ({round( ((epoch)/(epochs-1))*100 , 2)})%", decimals=2, length=70, empty=' ') if self.verbose==2 else range(0, len(features), self.batchsize):
        
        # main batching loop - iterate through the batches
        for batch_index in range(self.batchsize):

          if base_index + batch_index >= len(features):
            continue

          activations, weighted_sums = Propagate(features[base_index + batch_index])
          
          # if its a RNN model, append 0s to the 'disabled' layers
          # to match pred size with model size
          if self.RNN or self.LSTM or self.GRU:
            
            skibidi = 0
            target = []
            
            for layer in self.layers:
              if layer.return_output == False:
                target.append(0)
              else:
                if skibidi >= len(targets[base_index + batch_index]):
                  raise IndexError("not enough elements in the targets to verify")
                target.append(targets[base_index + batch_index][skibidi])
                skibidi += 1
          
          else:
            target = targets[base_index + batch_index]
          
          errors.append(Backpropagate(activations, weighted_sums, target))
          
          # errors.append([[0 for neuron in layer.neurons] for layer in self.layers])

          if self.RNN or self.LSTM or self.GRU:
            predicted = self.push(features[base_index + batch_index])

          else:
            predicted = activations[-1]
          
          epoch_loss += Key.ERROR[self.loss](targets[base_index + batch_index], predicted)
        
        ########################## update
        
        timestep += 1
        
        if 'frozenTS' in self.experimental:
          timestep = 1
        
        update(activations, weighted_sums, errors, timestep)
        errors = []

      # post-training stuff
      
      # log the eror
      epoch_loss /= len(features)
      self.error_logs.append(epoch_loss)
      
      # validation
      if len(validation_features) > 0:
        validation_loss = 0
        predicted_values = []
        
        # fetch the predictions
        for feature in validation_features:
          
          if self.RNN or self.LSTM or self.GRU:
            predicted = self.push(feature)

          else:
            predicted, _ = Propagate(feature)
            predicted = predicted[-1]
          
          predicted_values.append(predicted)
        
        if self.validation in Key.ERROR:
          for true, pred in zip(validation_targets, predicted_values):
          
            validation_loss += Key.ERROR[self.loss](true, pred)
          
          validation_loss /= len(validation_features)
          
        else:
          validation_loss = Key.METRICS[self.validation](predicted_values, validation_targets)
        
        self.validation_error_logs.append(validation_loss)
        
      # early stopping      
      if self.stopping and epoch >= self.patience and len(validation_features) > 0:
        
        patience_scope = self.validation_error_logs[ epoch - self.patience:epoch ]
        
        if all(validation_loss > err for err in patience_scope):
          break
      
      if epoch % self.logging == 0 and self.verbose>=3:
        
        prefix              = f"Epoch {epoch+1 if epoch == 0 else epoch}/{epochs-1} ({round( ((epoch+1)/epochs)*100 , 2)}%)"
        pad                 = ' ' * ( len(f"Epoch {epochs}/{epochs-1} (100.0%)") - len(prefix))
        suffix              = f" | Loss: {str(epoch_loss):22}"
        rate                = f" | ROC: {str(epoch_loss - self.error_logs[epoch-self.logging] if epoch >= self.logging else 0):23}"
        
        if len(validation_features) > 0:
          validation_suffix = f" | Validation: {str(validation_loss):22}"
          validation_rate   = f" | VROC: {validation_loss - self.validation_error_logs[epoch-self.logging] if epoch >= self.logging else 0}"
        
        else:
          validation_suffix = f" | Validation: {str('Unavailable'):12}"
          validation_rate   = f" | VROC: Unavailable"
        
        if 'track metrics' in self.experimental:
          
          Tpreds = []
          Vpreds = []
          
          for feature in features:
            Tpreds.append(self.push(feature))
          
          for validation in validation_features:
            Vpreds.append(self.push(validation))

          print(prefix + pad + suffix + validation_suffix + f"| Accuracy: {Key.METRICS['accuracy'](Tpreds, targets)}% | Validation Accuracy: {Key.METRICS['accuracy'](Vpreds, targets)}%")
          
        else:
          
          if self.verbose == 3:
            print(prefix + pad + suffix)
          elif self.verbose == 4:
            print(prefix + pad + suffix + rate)
          elif self.verbose == 5:
            print(prefix + pad + suffix + rate + validation_suffix)
          elif self.verbose == 6:
            print(prefix + pad + suffix + rate + validation_suffix + validation_rate)

  # post-processing

  def evaluate(self, features, targets, **kwargs) -> None:
    """
    Evaluate
    -----
      Validates the model based on the given validation data and prints out the results. this assumes an already compiled model.
      the results from this will automatically be saved to the summary table.
    -----
    Args
    -----
    - features           (list) : the features to use
    - targets            (list) : the corresponding targets to use
    - (optional) verbose (int)  : wether to show anything on screen
    - (optional) logging (bool) : wether to print out results after evaluating the model
    """
    self.is_validated = True
    
    verbose = kwargs.get('verbose', 0)
    logging = kwargs.get('logging', False)

    results = []
    longest = 0
    
    for metric_index in (utility.progress_bar(range(len(self.metrics)), f"Evaluating model", "Complete", decimals=2, length=75, empty=' ') if verbose == 1 else range(len(self.metrics))):

      metric = self.metrics[metric_index]
      correct = 0
      predicted = []
      longest = len(metric) if len(metric) > longest else longest
      
      for i in (utility.progress_bar(range(len(features)), f"Evaluating with {metric}", "Complete", decimals=2, length=75, empty=' ') if verbose == 2 else range(len(features))):

        predicted.append(self.push(features[i]))

      if metric in Key.METRICS:
        results.append(Key.METRICS[metric](predicted, targets))
      else:
        results.append(Key.ERROR[metric](targets, predicted))
    
    if logging:
      print("Evaluation Summary:")
      for metric, result in zip(self.metrics, results):
        
        pad = ' ' * (longest - len(metric))
        print(f"{pad}{metric} | {result}%" if metric in ('accuracy', 'precision', 'recall') else f"{pad}{metric} | {result}")

    print()    

  def push(self, x):
    """
    Propagate
    -----
      Propagates the input through the entire model, excluding dropout layers (if any).
      weights will not be updated.
    -----
    Args
    -----
      x (list) : the input to the model

    Returns
    -----
      output (list) : the output of the model, always a list
    """
    
    if type(x) != list:
      raise TypeError("""
    even though its not really a problem, the input to the model must be a list for consistency.
    navigate to the systems config to disable this error message.
    """)

    if self.RNN:

      output = 0      
      input_index = 0
      answer = []
      
      for layer in self.layers:

        if layer.accept_input:
          output = layer.apply(x[input_index], output)
          input_index += 1
          
        else:
          output = layer.apply(0, output)

        if layer.return_output:
          answer.append(output)
      
      return answer
      
    elif self.LSTM:
      
      long_term = 0
      short_term = 0
      input_index = 0
      answer = []
      
      for layer in self.layers:

        if layer.accept_input:
          long_term, short_term = layer.apply(x[input_index], long_term, short_term)
          input_index += 1
          
        else:
          long_term, short_term = layer.apply(0, long_term, short_term)

        if layer.return_output:
          answer.append(short_term)
      
      return answer
    
    elif self.GRU:
      
      output = 0
      input_index = 0
      answer = []
      
      for layer in self.layers:

        if layer.accept_input:
          output = layer.apply(x[input_index], output)
          input_index += 1
          
        else:
          output = layer.apply(0, output)

        if layer.return_output:
          answer.append(output)
      
      return answer
    
    else:
    
      for layer in self.layers:
        
        if type(layer) not in (Convolution, Dense, Maxpooling, Meanpooling, Flatten, Reshape, AFS, Operation, Localunit, RecurrentBlock, Dropout):
          raise TypeError(f"Unknown layer type {type(layer)}")

        elif type(layer) == Dropout:
          pass

        else:
          x = layer.apply(x)

      return x

  def peek(self) -> None:
    """
    Peek
    -----
      returns all the weights and biases of all the layers
    """

    for layer in range(len(self.layers)):
      print(f"layer {layer+1}")

      if type(self.layers[layer]) == Convolution:
        visual.numerical_display(self.layers[layer].kernel, title=f'Kernel bias: {self.layers[layer].bias}')
      elif type(self.layers[layer]) in (Dense, AFS, Localunit):
        for index, neuron in enumerate(self.layers[layer].neurons):
          print("  │")
          print(f"  ├─ Neuron  {index+1}")
          print(f"  │  Weights {neuron['weights']}")
          print(f"  │  Bias    {neuron['bias']}")
        print("  │")
        print(f"  └─ Activation Function: {self.layers[layer].activation}")
      else:
        print("  └─ No weights to peek")
      print()

  def summary(self, **kwargs) -> None:
    """
    Summary
    """

    counter = 0
    traniable_params = 0
    non_traniable_params = 0

    print("| Model Summary:")
    print("|")
    print(f"|   Status    : {'Uncompiled' if self.optimizer is None or self.loss is None else 'Compiled':25} | Training   : {'Untrained!' if self.is_trained==False else 'Trained':15}")
    print(f"|   Optimizer : {str(self.optimizer):25} | Validation : {'Validated' if self.is_validated else 'Unvalidated'}")
    print(f"|   Loss      : {str(self.loss):25} | Final Loss : {self.error_logs[-1]:15} ")
    print(f"|   Metrics   : {', '.join(self.metrics)}")
    print("|")
    print("| Model Overview:")
    print("|")
    print(f"|   Layer No.  | Layer Type  | Learnable | Input Shape | Output Shape | Layer Name")
    print(f"|              |             |           |             |              |")

    for layer in self.layers:
      counter += 1

      if type(layer) in (Dense, AFS, Localunit):
        traniable_params += len(layer.neurons) * len(layer.neurons[0]['weights']) + 1 if layer.learnable else 0
        non_traniable_params += len(layer.neurons) * len(layer.neurons[0]['weights']) + 1 if not layer.learnable else 0

      elif type(layer) == Convolution:
        traniable_params += len(layer.kernel) * len(layer.kernel[0]) + 1 if layer.learnable else 0
        non_traniable_params += len(layer.kernel) * len(layer.kernel[0]) + 1 if not layer.learnable else 0

      if type(layer) in (Dense, Convolution, AFS, Localunit):
        learnable = True if layer.learnable else False
        print(f"|   {counter:10} | {layer.__class__.__name__:11} | {str(learnable):10}| {str(self.sizes[counter-1]):11} | {str(self.sizes[counter]):12} | {layer.name:20}")

      else:
        print(f"|   {counter:10} | {layer.__class__.__name__:11} |           | {str(self.sizes[counter-1]):11} | {str(self.sizes[counter]):12} | {layer.name:20}")

    print("|")
    print("| Parameter Breakdown:")
    print("|")
    print(f"|   Trainable Parameters     : {traniable_params}")
    print(f"|   Non-Trainable Parameters : {non_traniable_params}")
    print(f"|   Total Parameters         : {traniable_params + non_traniable_params}")
    print()

#######################################################################################################
#                                           Parallel Class                                            #
#######################################################################################################

class RecurrentBlock:
  # pre-processing

  def __init__(self, *args):
    """
    Recurrent Block
    ======
      Enables merging Recurrent layers with non-recurrent layers
    -----

    Compatible layers:
    - Recurrent
    - LSTM
    - GRU
    
    Refer to the documentation for more information.
    """
    # make sure all the layers are alike
    self.layers = args if args != [] else None

    # defined during compiling
    self.optimizer      = None
    self.loss           = None
    self.learning_rate  = None
    
    self.activations    = None
    self.WS             = None
    self.errors         = []

    self.optimizer_instance = Optimizer()

  # executed once
  def calibrate(self, optimizer, loss, learning_rate, **kwargs):
    """
    Internal function, do not invoke
    """
    self.optimizer = optimizer.lower()
    self.loss = loss.lower()
    self.learning_rate = learning_rate

    self.alpha = kwargs.get('alpha', None)
    self.beta = kwargs.get('beta', None)
    self.epsilon = kwargs.get('epsilon', None) # zerodivison prevention
    self.gamma = kwargs.get('gamma', None)
    self.delta = kwargs.get('delta', None)

    self.experimental = kwargs.get('experimental', [])
    
    self.RNN  = any(type(layer) == Recurrent for layer in self.layers)
    self.LSTM = any(type(layer) == LSTM for layer in self.layers)
    self.GRU  = any(type(layer) == GRU for layer in self.layers)
    
    # formatting check
    for branch in self.layers:
      
      if type(branch) not in (Recurrent, LSTM, GRU):
        raise RecursionError(f"RecurrentBlock only accepts Recurrent units and not '{branch.__class__.__name__}'")
      
      if type(branch) != type(self.layers[0]):
        raise TypeError("All branches must be of the same type")

  def internal(self, procedure, *args, **kwargs):
    """
    Internal class, do not invoke
    """
    
    def Propagate(input):

      activations = []
      weighted_sums = []

      if self.RNN:
        
        input_index = 0
        output = 0
        outputWS = 0
        for layer in self.layers: # get activations
          
          if layer.accept_input:
            
            output = layer.apply(x[input_index], output)
            
            activations.append([input[input_index], output])
            
            outputWS = layer.get_weighted_sum(x[input_index], output)
            
            input_index += 1
            
          else:
            output = layer.apply(0, output)
            activations.append([0, output])
            outputWS = layer.get_weighted_sum(0, output)
          
          weighted_sums.append(outputWS)
      
      elif self.LSTM:
        
        long_memory = 0
        short_memory = 0
        input_index = 0
        
        for layer in self.layers: # get activations
          
          if layer.accept_input:
            long_memory, short_memory = layer.apply(input[input_index], long_memory, short_memory)
            activations.append([input[input_index], long_memory, short_memory])
            LT, ST, final_long_term, final_short_term, merged_state, gated_long_term, weighted_input, weighted_short_term = layer.get_weighted_sum(input[input_index], long_memory, short_memory)
            input_index += 1
          
          else:
            long_memory, short_memory = layer.apply(0, long_memory, short_memory)
            activations.append([0, long_memory, short_memory])
            LT, ST, final_long_term, final_short_term, merged_state, gated_long_term, weighted_input, weighted_short_term = layer.get_weighted_sum(0, long_memory, short_memory)
          
          weighted_sums.append([LT, ST, final_long_term, final_short_term, merged_state, gated_long_term, weighted_input, weighted_short_term])
      
      elif self.GRU:
          
        input_index = 0
        output = 0
        outputWS = 0
        for layer in self.layers: # get activations
          
          if layer.accept_input:
            
            output = layer.apply(x[input_index], output)
            
            activations.append([input[input_index], output])
            
            outputWS = layer.get_weighted_sum(x[input_index], output)
            
            input_index += 1
            
          else:
            output = layer.apply(0, output)
            activations.append([0, output])
            outputWS = layer.get_weighted_sum(0, output)
          
          weighted_sums.append(outputWS)
      
      self.activations = activations
      self.WS          = weighted_sums

    def Backpropagate(incoming_errors, **kwargs):
      # incoming errors is already splitted to delegate each branch its respective error
      
      activations   = self.activations
      weighted_sums = self.WS
      
      LSTM_incoming_input = kwargs.get('LSTM_incoming_input', [])
      RNN_input_err = []

      errors = [0] * len(self.layers)
      

      output_layer_errors = []

      if type(self.layers[-1]) == Recurrent: # if its a recurrent layer
        recurrent_output_errors = incoming_errors[:]
        
        error_C = incoming_errors[-1]
        error_D = 0
        total_error = error_C + error_D
        
        derivative = Key.ACTIVATION_DERIVATIVE[self.layers[-1].activation](weighted_sums[-1])
        
        error_Wa = derivative * total_error * activations[-1][0]
        error_Wb = derivative * total_error * activations[-1][1]
        error_B  = derivative * total_error * self.layers[-1].input_weight
        error_a  = derivative * total_error * self.layers[-1].carry_weight
        
        output_layer_errors = [error_Wa, error_Wb, error_B, error_a]
        RNN_input_err.append(error_B)
      
      elif type(self.layers[-1]) == LSTM: # if its a LSTM layer
        this_layer = self.layers[-1]
        
        L_error   = 0
        S_error   = 0
        out_error = incoming_errors[-1]
        
        incoming_ST, incoming_LT, final_long_term, final_short_term, merged_state, gated_long_term, weighted_input, weighted_short_term = weighted_sums[-1]
        incoming_input = incoming_errors[-1]
        
        ST_weights    = self.layers[-1].short_term_weights
        extra_weights = self.layers[-1].extra_weights
        input_weights = self.layers[-1].input_weights
        
        total_error = out_error + S_error
        
        # calculate OUT
        
        out_we = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term)  * extra_weights[0] * extra_weights[1] * total_error # calculate [we]
        out_a  = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term) * extra_weights[1] * total_error * extra_weights[2]  # calculate [a]
        out_b  = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term) * extra_weights[0] * total_error * extra_weights[2]  # calculate [b]
        
        A = Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term)            * extra_weights[0] * extra_weights[1] * extra_weights[2] * total_error
        B = Key.ACTIVATION["sigmoid"](merged_state[3])            * Key.ACTIVATION_DERIVATIVE["tanh"](final_long_term) * extra_weights[0] * extra_weights[1] * extra_weights[2] * total_error + L_error
        
        # calculate OUT
        
        merged_D = 1 * A 
        short_D = incoming_ST * A
        input_D = incoming_input * A
        
        # calculate INPUT GATE
        
        B_gate_error = B * Key.ACTIVATION["tanh"](merged_state[2])
        C_gate_error = B * Key.ACTIVATION["sigmoid"](merged_state[1])
        
        merged_B = 1 * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) *  B_gate_error
        merged_C = 1 * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2])    * C_gate_error
        
        input_B = incoming_input * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error
        short_B = incoming_ST    * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error
        
        input_C = incoming_input * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * C_gate_error
        short_C = incoming_ST    * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * C_gate_error
        
        # calculate FORGET GATE
        
        merged_A = 1              * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B
        short_A  = incoming_ST    * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B
        input_A  = incoming_input * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B
        
        # calculate CARRY
        
        carry_ST = (ST_weights[0] * A) + \
                    (ST_weights[1] * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error) + \
                    (ST_weights[2] * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2])    * C_gate_error) + \
                    (ST_weights[3] * A)
        
        carry_LT = B * Key.ACTIVATION["sigmoid"](merged_state[0])
        
        RNN_input_err.append(
            (input_weights[0] * A ) + \
            (input_weights[1] * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error ) + \
            (input_weights[2] * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2])    * C_gate_error ) + \
            (input_weights[3] * A )
          )
        
        output_layer_errors = [carry_LT, carry_ST, 
                                [merged_A, merged_B, merged_C, merged_D], 
                                [short_A, short_B, short_C, short_D], 
                                [input_A, input_B, input_C, input_D], 
                                [out_a, out_b, out_we]
                              ]
        
      elif type(self.layers[-1]) == GRU: # if its a GRU layer
        this_layer  = self.layers[-1]
        carry_error = 0
        out_error   = incoming_errors[-1]
        
        input_weights = this_layer.input_weights
        carry_weights = this_layer.carry_weights
        
        total_error = out_error + carry_error
        final_output, gated_carry, merged_state, weighted_input, weighted_carry, incoming_input, incoming_carry = weighted_sums[-1]
        
        # output gate
        
        error_C = Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * ( 1 - Key.ACTIVATION["sigmoid"](merged_state[1]) ) * total_error
        
        bias_C  = 1                                                             * error_C
        input_C = incoming_input                                                * error_C
        carry_C = (Key.ACTIVATION["sigmoid"](merged_state[0]) * incoming_carry) * error_C

        # update gate
        
        error_B = total_error * (incoming_carry - Key.ACTIVATION["tanh"](merged_state[2])) * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1])
                  
        bias_B  = 1              * error_B
        input_B = incoming_input * error_B
        carry_B = incoming_carry * error_B
        
        # reset gate
        
        error_A = total_error * (1 - Key.ACTIVATION["sigmoid"](merged_state[1])) * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * incoming_carry * carry_weights[2] * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0])

        bias_A  = 1              * error_A
        input_A = incoming_input * error_A
        carry_A = incoming_carry * error_A
        
        # calculate upstream gradient
        
        carry_error = ( carry_weights[0] * error_A                                                                                                                                                          ) + \
                      ( carry_weights[1] * error_B                                                                                                                                                          ) + \
                      ( total_error * (1 - Key.ACTIVATION["sigmoid"](merged_state[1])) * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * Key.ACTIVATION["sigmoid"](merged_state[0]) * carry_weights[2] ) + \
                      ( total_error * Key.ACTIVATION["sigmoid"](merged_state[1])                                                                                                                            ) 
        
        RNN_input_err.append(
            (input_weights[0] * error_A) + \
            (input_weights[1] * error_B) + \
            (input_weights[2] * error_C)
          )
        
        output_layer_errors = [
                                carry_error, 
                                [bias_A, bias_B, bias_C], 
                                [carry_A, carry_B, carry_C], 
                                [input_A, input_B, input_C]
                              ]
        
      errors[-1] = output_layer_errors
    
    # if its a RNN, LSTM or GRU layer
      for branch_index in reversed(range(len(self.layers)-1)):
        # FRONT | next layer --> this layer --> previous layer | BACK
        # dont forget that this is a parralel class.

        branch = self.layers[branch_index]
        this_activations = activations[branch_index]
        this_WS          = weighted_sums[branch_index]

        prev_errors = errors[branch_index - 1]

        layer_errors = []

        if type(branch) == Recurrent:
          
          error_C = recurrent_output_errors[branch_index]
          error_D = prev_errors[3]
          
          if branch.return_output:
            total_error = error_C + error_D
          else:
            total_error = error_D
          
          derivative = Key.ACTIVATION_DERIVATIVE[branch.activation](this_WS)
          
          error_Wa = derivative * total_error * this_activations[0]
          error_Wb = derivative * total_error * this_activations[1]
          error_B  = derivative * total_error * branch.input_weight
          error_a  = derivative * total_error * branch.carry_weight
          
          layer_errors = [error_Wa, error_Wb, error_B, error_a]
          RNN_input_err.append(error_B)
        
        elif type(branch) == LSTM:
          
          L_error   = prev_errors[0]
          S_error   = prev_errors[1]
          out_error = incoming_errors[branch_index] if this_layer.return_output else 0
          
          incoming_ST, incoming_LT, final_long_term, final_short_term, merged_state, gated_long_term, weighted_input, weighted_short_term = weighted_sums[branch_index]
          incoming_input = LSTM_incoming_input[branch_index]
          
          ST_weights    = branch.short_term_weights
          extra_weights = branch.extra_weights
          input_weights = branch.input_weights
          
          total_error = out_error + S_error
          
          # calculate OUT
          
          out_we = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term)  * extra_weights[0] * extra_weights[1] * total_error # calculate [we]
          out_a  = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term) * extra_weights[1] * total_error * extra_weights[2]  # calculate [a]
          out_b  = Key.ACTIVATION["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term) * extra_weights[0] * total_error * extra_weights[2]  # calculate [b]
          
          A = Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[3]) * Key.ACTIVATION["tanh"](final_long_term)            * extra_weights[0] * extra_weights[1] * extra_weights[2] * total_error
          B = Key.ACTIVATION["sigmoid"](merged_state[3])            * Key.ACTIVATION_DERIVATIVE["tanh"](final_long_term) * extra_weights[0] * extra_weights[1] * extra_weights[2] * total_error + L_error
          
          # calculate OUT
          
          merged_D = 1 * A 
          short_D = incoming_ST * A
          input_D = incoming_input * A
          
          # calculate INPUT GATE
          
          B_gate_error = B * Key.ACTIVATION["tanh"](merged_state[2])
          C_gate_error = B * Key.ACTIVATION["sigmoid"](merged_state[1])
          
          merged_B = 1 * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error
          merged_C = 1 * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2])    * C_gate_error
          
          input_B = incoming_input * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error
          short_B = incoming_ST    * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error
          
          input_C = incoming_input * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * C_gate_error
          short_C = incoming_ST    * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * C_gate_error
          
          # calculate FORGET GATE
          
          merged_A = 1              * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B 
          short_A  = incoming_ST    * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B
          input_A  = incoming_input * incoming_LT * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0]) * B
          
          # calculate CARRY
          
          carry_ST = (ST_weights[0] * A) + \
                      (ST_weights[1] * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error) + \
                      (ST_weights[2] * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2])    * C_gate_error) + \
                      (ST_weights[3] * A)
          
          carry_LT = B * Key.ACTIVATION["sigmoid"](merged_state[0])
          
          RNN_input_err.append(
            (input_weights[0] * A ) + \
            (input_weights[1] * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1]) * B_gate_error ) + \
            (input_weights[2] * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2])    * C_gate_error ) + \
            (input_weights[3] * A )
          )
          
          layer_errors = [carry_LT, carry_ST, 
                          [merged_A, merged_B, merged_C, merged_D], 
                          [short_A, short_B, short_C, short_D], 
                          [input_A, input_B, input_C, input_D], 
                          [out_a, out_b, out_we]
                          ]
          
        elif type(branch) == GRU:

          carry_error = prev_errors[0]
          out_error   = incoming_errors[branch_index]
          
          input_weights = this_layer.input_weights
          carry_weights = this_layer.carry_weights
          
          total_error = out_error + carry_error
          final_output, gated_carry, merged_state, weighted_input, weighted_carry, incoming_input, incoming_carry = weighted_sums[branch_index]
          
          # output gate
          
          error_C = Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * ( 1 - Key.ACTIVATION["sigmoid"](merged_state[1]) ) * total_error
          
          bias_C  = 1                                                             * error_C
          input_C = incoming_input                                                * error_C
          carry_C = (Key.ACTIVATION["sigmoid"](merged_state[0]) * incoming_carry) * error_C

          # update gate
          
          error_B = total_error * (incoming_carry - Key.ACTIVATION["tanh"](merged_state[2])) * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[1])
                    
          bias_B  = 1              * error_B
          input_B = incoming_input * error_B
          carry_B = incoming_carry * error_B
          
          # reset gate
          
          error_A = total_error * (1 - Key.ACTIVATION["sigmoid"](merged_state[1])) * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * incoming_carry * carry_weights[2] * Key.ACTIVATION_DERIVATIVE["sigmoid"](merged_state[0])

          bias_A  = 1              * error_A
          input_A = incoming_input * error_A
          carry_A = incoming_carry * error_A
          
          # calculate upstream gradient
          
          carry_error = ( carry_weights[0] * error_A                                                                                                                                                          ) + \
                        ( carry_weights[1] * error_B                                                                                                                                                          ) + \
                        ( total_error * (1 - Key.ACTIVATION["sigmoid"](merged_state[1])) * Key.ACTIVATION_DERIVATIVE["tanh"](merged_state[2]) * Key.ACTIVATION["sigmoid"](merged_state[0]) * carry_weights[2] ) + \
                        ( total_error * Key.ACTIVATION["sigmoid"](merged_state[1])                                                                                                                            ) 
          
          RNN_input_err.append(
              (input_weights[0] * error_A) + \
              (input_weights[1] * error_B) + \
              (input_weights[2] * error_C)
            )
          
          layer_errors = [
                          carry_error, 
                          [bias_A, bias_B, bias_C], 
                          [carry_A, carry_B, carry_C], 
                          [input_A, input_B, input_C]
                          ]
        
        errors[branch_index] = layer_errors[:]

      self.errors = errors
      return RNN_input_err
      
    def update(learning_rate, timestep, batchsize):
      errors = self.errors[:]
      
      alpha = self.alpha
      beta = self.beta
      epsilon = self.epsilon
      gamma = self.gamma
      delta = self.delta

      optimize = Key.OPTIMIZER.get(self.optimizer)

      param_id = 0 # must be a positive integer

      for branch_index, branch in enumerate(self.layers):
        
        if type(branch) == Recurrent and branch.learnable:
          Wa_gradient = 0
          Wb_gradient = 0
          B_gradient  = 0
          
          for error_version in errors:
            
            Wa_gradient += error_version[branch_index][0]
            Wb_gradient += error_version[branch_index][1]
            B_gradient  += error_version[branch_index][2]

          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            Wa_gradient /= batchsize
            Wb_gradient /= batchsize
            B_gradient  /= batchsize

          param_id += 1
          branch.carry_weight = optimize(learning_rate, branch.carry_weight, Wa_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
          param_id += 1
          branch.input_weight = optimize(learning_rate, branch.input_weight, Wb_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
          param_id += 1
          branch.bias         = optimize(learning_rate, branch.bias, B_gradient, self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
        
        elif type(branch) == LSTM and branch.learnable:
          
          B_ERR     = [0] * 4 # biases (merge)
          ST_ERR    = [0] * 4 # short term weights
          INPUT_ERR = [0] * 4 # input weights
          EXTRA_ERR = [0] * 3 # extra weights (output gate)
          
          for error_version in errors:
            B_ERR     = [B_ERR[i]     + error_version[branch_index][2][i] for i in range(4)]
            ST_ERR    = [ST_ERR[i]    + error_version[branch_index][3][i] for i in range(4)]
            INPUT_ERR = [INPUT_ERR[i] + error_version[branch_index][4][i] for i in range(4)]
            EXTRA_ERR = [EXTRA_ERR[i] + error_version[branch_index][5][i] for i in range(3)]
            
          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            B_ERR     = [a / batchsize for a in B_ERR]
            ST_ERR    = [b / batchsize for b in ST_ERR]
            INPUT_ERR = [c / batchsize for c in INPUT_ERR]
            EXTRA_ERR = [d / batchsize for d in EXTRA_ERR]
          
          for index, bias in enumerate(branch.biases):
            param_id += 1
            branch.biases[index] = optimize(learning_rate, bias, B_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
          
          for index, weight in enumerate(branch.short_term_weights):
            param_id += 1
            branch.short_term_weights[index] = optimize(learning_rate, weight, ST_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
          
          for index, weight in enumerate(branch.input_weights):
            param_id += 1
            branch.input_weights[index] = optimize(learning_rate, weight, INPUT_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
          
          for index, weight in enumerate(branch.extra_weights):
            param_id += 1
            if branch.version == 'statquest':
              branch.extra_weights[index] = optimize(learning_rate, weight, EXTRA_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)
        
        elif type(branch) == GRU and branch.learnable:
          
          B_ERR     = [0] * 3 # biases
          C_ERR     = [0] * 3 # short term weights
          INPUT_ERR = [0] * 3 # input weights
          
          for error_version in errors:
            B_ERR     = [B_ERR[i]     + error_version[branch_index][1][i] for i in range(3)]
            C_ERR     = [C_ERR[i]     + error_version[branch_index][2][i] for i in range(3)]
            INPUT_ERR = [INPUT_ERR[i] + error_version[branch_index][3][i] for i in range(3)]
            
          if (self.optimizer != 'none') and ('fullgrad' not in self.experimental):
            B_ERR     = [a / batchsize for a in B_ERR]
            C_ERR     = [b / batchsize for b in C_ERR]
            INPUT_ERR = [c / batchsize for c in INPUT_ERR]
          
          for index, bias in enumerate(branch.biases):
            param_id += 1
            branch.biases[index] = optimize(learning_rate, bias, B_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

          for index, weight in enumerate(branch.carry_weights):
            param_id += 1
            branch.carry_weights[index] = optimize(learning_rate, weight, C_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

          for index, weight in enumerate(branch.input_weights):
            param_id += 1
            branch.input_weights[index] = optimize(learning_rate, weight, INPUT_ERR[index], self.optimizer_instance, alpha=alpha, beta=beta, epsilon=epsilon, gamma=gamma, delta=delta, param_id=param_id, timestep=timestep)

    if procedure == 1:
      Propagate(*args)
    
    elif procedure == 2:
      return Backpropagate(*args, **kwargs)
    
    elif procedure == 3:
      update(*args, **kwargs)
    
  def apply(self, x):
    """
    Args
    -----
    x (list) : the input to the model
    """

    if self.RNN:

      output = 0      
      input_index = 0
      answer = []
      
      for layer in self.layers:

        if layer.accept_input:
          output = layer.apply(x[input_index], output)
          input_index += 1
          
        else:
          output = layer.apply(0, output)

        if layer.return_output:
          answer.append(output)
      
      return answer
      
    elif self.LSTM:
      
      long_term = 0
      short_term = 0
      input_index = 0
      answer = []
      
      for layer in self.layers:

        if layer.accept_input:
          long_term, short_term = layer.apply(x[input_index], long_term, short_term)
          input_index += 1
          
        else:
          long_term, short_term = layer.apply(0, long_term, short_term)

        if layer.return_output:
          answer.append(short_term)
      
      return answer
    
    elif self.GRU:
      
      output = 0
      input_index = 0
      answer = []
      
      for layer in self.layers:

        if layer.accept_input:
          output = layer.apply(x[input_index], output)
          input_index += 1
          
        else:
          output = layer.apply(0, output)

        if layer.return_output:
          answer.append(output)
      
      return answer
    
    else:
      raise ValueError("Internal attribute not set.")
    
#######################################################################################################
#                                         Component Classes                                           #
#######################################################################################################

# learnable layers

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
    - ELU
    - GELU
    - SELU
    - ReEU
    - None
    - TANDIP

    Normalization functions
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    """
    self.kernel_shape = kernel
    self.channels = channels
    
    self.activation = activation.lower()
    self.learnable = kwargs.get('learnable', True)
    self.use_bias = kwargs.get('bias', True)
    self.bias = [0]
    self.name = kwargs.get('name', 'convolution')
    self.input_shape = 0

  def reshape_input_shape(self, input_shape):
    self.input_shape = input_shape
    
    self.kernels = arraytools.generate_random_array(self.kernel_shape[0], self.kernel_shape[1], input_shape[2], self.channels)
    self.bias = arraytools.generate_random_array(input_shape[2], self.channels) if self.use_bias else arraytools.generate_array(input_shape[2], self.channels, value=0)
    
  def apply(self, multichannel_input):
    
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
        
            output[i][j] = Key.ACTIVATION[self.activation](dot_product) + bias 

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

class Dense:
  def __init__(self, neurons:int, activation:str, **kwargs):
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
    - ELU
    - GELU
    - SELU
    - ReEU
    - None
    - TANDIP

    Normalization functions
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
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
    
    if activation.lower() in ('relu','softplus','mish','swish','leaky relu','elu','gelu','selu','reeu','none','tandip'):
      default_initializer = 'he normal'
    elif activation.lower() in ('binary step','softsign','sigmoid','tanh'):
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
      'bias': Key.INITIALIZATION[self.initialization](input_shape, output_shape)
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
        sum(input_val * weight_val for input_val, weight_val in zip(input, _neuron['weights'])) + _neuron['bias']
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
  def __init__(self, receptive_field:int, activation:str, **kwargs):
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
    - ELU
    - GELU
    - SELU
    - ReEU
    - None
    - TANDIP

    Normalization functions
    - Binary Step
    - Softsign
    - Sigmoid
    - Tanh
    
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
    self.activation = activation
    self.name = kwargs.get('name', 'local unit')
    self.learnable = kwargs.get('learnable', True)
    
    if activation.lower() in ('relu','softplus','mish','swish','leaky relu','elu','gelu','selu','reeu','none','tandip'):
      default_initializer = 'he normal'
    elif activation.lower() in ('binary step','softsign','sigmoid','tanh'):
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
      'bias': Key.INITIALIZATION[self.initialization](input_shape, next_shape)
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

      answer.append(Key.ACTIVATION[self.activation](dot_product + self.neurons[a]['bias']))
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

class AFS:
  def __init__(self, activation:str, **kwargs):
    """
    Adaptive Feature Scaler
    -----
      Experimental layer, use a 'None' activation function for a traditional AFS layer, else its no longer
      a scaler layer anymore.
    -----
    Args
    -----
    - activation           (string)  : the activation function to use for the attention layer
    - (Optional) bias      (boolean) : whether or not to use bias
    - (Optional) learnable (boolean) : whether or not to learn
    - (Optional) name      (string)  : the name of the layer
    -----
      (Activation functions)
      - ReLU
      - Softplus
      - Mish
      - Swish
      - Leaky ReLU
      - ELU
      - GELU
      - SELU
      - ReEU
      - None
      - TANDIP

      (Normalization functions)
      - Binary Step
      - Softsign
      - Sigmoid
      - Tanh
      
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
    self.activation = activation.lower()
    self.name = kwargs.get('name', 'feature scaler')
    self.learnable = kwargs.get('learnable', True)
    self.use_bias = kwargs.get('bias', True)
    
    if activation.lower() in ('relu','softplus','mish','swish','leaky relu','elu','gelu','selu','reeu','none','tandip'):
      default_initializer = 'he normal'
    elif activation.lower() in ('binary step','softsign','sigmoid','tanh'):
      default_initializer = 'glorot normal'
    else:
      raise ValueError(f"Unknown activation function: '{activation}'")
    
    self.initialization = kwargs.get('initialization', default_initializer).lower()

    neuron = {
      'weight': 0,
      'bias': 0
      }
    self.neurons = [neuron]

  def reshape_input_shape(self, input_shape, output_shape):

    self.neurons = [
      {
      'weight': Key.INITIALIZATION[self.initialization](input_shape, output_shape),
      'bias': Key.INITIALIZATION[self.initialization](input_shape, output_shape)
      }
      for _ in range(input_shape)
    ]

  def apply(self, input):

    return [Key.ACTIVATION[self.activation](self.neurons[i]['weight'] * input[i] + self.neurons[i]['bias']) for i in range(len(input))]

  def get_weighted_sum(self, input: list):
    self.input = input[:]
    answer = []

    if type(input) != list:
      raise TypeError("input must be a 1D array list")
    if type(input[0]) not in (int, float):
      raise TypeError("input must be a 1D array list \nuse the built-in 'Flatten' layer before an AFS layer")

    return [self.neurons[i]['weight'] * input[i] + self.neurons[i]['bias'] for i in range(len(input))]

class Recurrent:
  def __init__(self, activation:str, **kwargs):
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
    self.activation = activation
    
    self.accept_input = kwargs.get('input', True)
    self.return_output = kwargs.get('output', True)
    
    self.name = kwargs.get('name', 'recurrent')
    self.learnable = kwargs.get('learnable', True)
    
    self.input_weight = random.uniform(0.1, 1)
    self.carry_weight = random.uniform(0.1, 1)
    self.bias = random.uniform(-0.5, 0.5)
  
  def apply(self, input, carry):
    
    return Key.ACTIVATION[self.activation]((input * self.input_weight) + (carry * self.carry_weight) + self.bias)

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
    - activation           (string)  : the activation function to use for the attention layer
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
    self.input_size = arraytools.shape(input)

    for input in input_channels:
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
    self.input_size = arraytools.shape(input)

    for input in input_channels:
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
      multichannel_ans.appen(answer)

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
    return arraytools.reshape(input, self.width, self.height)

class Operation:
  def __init__(self, operation, **kwargs):
    """
    Operation
    -----
      Operational layer for functions or normalizations.
    -----
    Args
    -----
    - operation    (string) : the scaler to use
    - (Optional) name   (string) : the name of the layer
    -----
    Available Operations:

      (Scalers)
      - standard scaler
      - min max scaler
      - max abs scaler
      - robust scaler

      (Activation functions)
      - ReLU
      - Softplus
      - Mish
      - Swish
      - Leaky ReLU
      - ELU
      - GELU
      - SELU
      - ReEU
      - None
      - TANDIP

      (Normalization functions)
      - Binary Step
      - Softsign
      - Sigmoid
      - Tanh
      - Softmax
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


