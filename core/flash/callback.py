
class Callback:
  """
  Callable
  -----
    A callback is a function that is called during the training process. It is advised to create a custom callback and inherit from this class since 
    there are some essential methods that need to be present.
  -----
  Args
  -----
  - __init__       (function) : a callback instance is created before training, no arguments are passed
  - initialization (function) : called only once after the callback instance is created
  - before_epoch   (function) : called once at the start of each epoch
  - before_update  (function) : called once before backpropagation and update
  - after_update   (function) : called once after backpropagation and update
  - after_epoch    (function) : called once at the end of each epoch
  - end            (function) : called once at the end of training
  """
  def __init__(callbackself):
    pass
  
  def initialization(callbackself, *args, **kwargs):
    pass
  
  def before_epoch(callbackself, *args, **kwargs):
    pass
  
  def before_update(callbackself, *args, **kwargs):
    pass
  
  def after_update(callbackself, *args, **kwargs):
    pass
  
  def after_epoch(callbackself, *args, **kwargs):
    pass
  
  def end(callbackself, *args, **kwargs):
    pass