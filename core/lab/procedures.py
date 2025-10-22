import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc import ABC, abstractmethod
import copy

# to impliment:
# - conditonals
# - checkpoints.

class Procedure(ABC):
  """
  Base class for procedures to be applied to a StandardNet model
  
  A Procedure class is required to have the following:
  
  - A '__init__' method with any constant object attributes should be defined here
    - Args:
      - model     (StandardNet model) : the model to apply the spesific procedure to, must be compiled
      - *args     (any)               : any arguments to be passed to the procedure
      - **kwargs  (any)               : any keyword arguments to be passed to the procedure
    - Returns:
      - None
  
  - A '__call__' method that applies the procedure to the model
    - Args:
      - iteration (int)               : the current cycle of the experiment
      - *args     (any)               : any arguments to be passed to the procedure
      - **kwargs  (any)               : any keyword arguments to be passed to the procedure
    - Returns:
      - dict : anything that needs to be logged
  """

  @abstractmethod
  def __init__(self, model, *args, **kwargs):
    self.model = model
    self.args = args
    self.kwargs = kwargs
  
  @abstractmethod
  def __call__(self, timestep, *args, **kwargs):
    return {}

class Track_Gradients(Procedure):
  def __init__(self, model, datapath:tuple[str,str], *args, **kwargs):
    self.model = model
    self.datapath = datapath
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, iteration, *args, **kwargs):
    gradients = self.model.gradients_history[self.datapath[0]][self.datapath[1]]
    
    return {"gradients":gradients}

class Track_Parameters(Procedure):
  def __init__(self, model, datapath:tuple[str,str], *args, **kwargs):
    self.model = model
    self.datapath = datapath
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, iteration, *args, **kwargs):
    parameters = self.model.params_history[self.datapath[0]][self.datapath[1]]
    return {"parameters":parameters}

class Ablate(Procedure):
  def __init__(self, model, targer_layer_indecies:tuple[int,...], *args, **kwargs):
    """
    Standardnet:recompile
    Staticnet:ok
    """
    self.model = model 
    self.model_layers = model.layers # immutable
    self.indecises = targer_layer_indecies
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, timestep, *args, **kwargs):
    temporary_layers = copy.deepcopy(self.model_layers)
    
    temporary_layers.pop( self.indecises[timestep % len(self.indecises)] )
    self.model.layers = temporary_layers

    return {"ablated layer": self.model.layers[self.indecises[timestep % len(self.indecises)]].__class__.__name__}
    
