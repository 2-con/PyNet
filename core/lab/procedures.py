import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from abc import ABC, abstractmethod

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
      - model     (StandardNet model) : the model to apply the spesific procedure to, must be compiled
      - *args     (any)               : any arguments to be passed to the procedure
      - **kwargs  (any)               : any keyword arguments to be passed to the procedure
    - Returns:
      - whatever the procedure returns
  """

  @abstractmethod
  def __init__(self, model, *args, **kwargs):
    self.model = model
    self.args = args
    self.kwargs = kwargs
  
  @abstractmethod
  def __call__(self, model, *args, **kwargs):
    pass

class Execute(Procedure):
  def __call__(self, *args, **kwargs):
    if not hasattr(self.model, self.method):
      raise AttributeError(f"Model does not have method {self.method}")
    
    return getattr(self.model, self.method)(*self.args, **self.kwargs)

class Track_Gradients(Procedure):
  def __init__(self, model, datapath:tuple[str, str], *args, **kwargs):
    self.model = model
    self.datapath = datapath
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, netlab_instance, *args, **kwargs):
    gradients = self.model.latest_gradients[self.datapath[0]][self.datapath[1]]
    
    netlab_instance.log(gradients)
    return gradients

class Track_Parameters(Procedure):
  def __init__(self, model, datapath:tuple[str, str], *args, **kwargs):
    self.model = model
    self.datapath = datapath
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, netlab_instance, *args, **kwargs):
    parameters = self.model.latest_parameters[self.datapath[0]][self.datapath[1]]
    
    netlab_instance.log(parameters)
    return parameters

class Ablate(Procedure):
  def __init__(self, model, target_layer_name:str, mode:str='freeze', *args, **kwargs):
    self.model = model
    self.target_layer_name = target_layer_name
    self.mode = mode
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, netlab_instance, *args, **kwargs):
    raise NotImplementedError("Ablation not yet implemented")

class Checkpoint(Procedure):
  def __init__(self, model, *args, **kwargs):
    self.model = model
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, netlab_instance, *args, **kwargs):
    raise NotImplementedError("Checkpoint not yet implemented")

class Conditional(Procedure):
  def __init__(self, model, condition, steps_to_run:list[Procedure], *args, **kwargs):
    self.model = model
    self.condition = condition
    self.steps_to_run = steps_to_run
    self.args = args
    self.kwargs = kwargs
  
  def __call__(self, netlab_instance, *args, **kwargs):
    if self.condition:
      results = []
      for step in self.steps_to_run:
        results.append(step(...))
      return results
    
    else:
      return None