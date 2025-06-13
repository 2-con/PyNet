import math

def Root_Mean_Squared_Error(y_true:list, y_pred:list):
  """
  Root Mean Squared Error
  -----
    Returns the root mean squared error of the model
  -----
  Args
  -----
  - y_true (list) : the true values
  - y_pred (list) : the predicted values

  Returns
  -----
    (float) : the root mean squared error of the model
  """
  return math.sqrt(Mean_squared_error(y_true, y_pred))

def Mean_squared_error(y_true:list, y_pred:list):
  if len(y_true) != len(y_pred):
    raise Exception("y_true and y_pred must have the same length")
  
  answer = 0

  for true, pred in zip(y_true, y_pred):
    answer += (true - pred)**2

  return (answer / len(y_true))/2

def Mean_absolute_error(y_true:list, y_pred:list):
  if len(y_true) != len(y_pred):
    raise Exception("y_true and y_pred must have the same length")
  
  answer = 0

  for true, pred in zip(y_true, y_pred):
    answer += abs(true - pred)

  return answer / len(y_true)

def Total_squared_error(y_true:list, y_pred:list):
  if len(y_true) != len(y_pred):
    raise Exception("y_true and y_pred must have the same length")
  
  answer = 0

  for true, pred in zip(y_true, y_pred):
    answer += (true - pred)**2

  return answer/2

def Total_absolute_error(y_true:list, y_pred:list):
  if len(y_true) != len(y_pred):
    raise Exception("y_true and y_pred must have the same length")
  
  answer = 0

  for true, pred in zip(y_true, y_pred):
    answer += abs(true - pred)

  return answer

def Categorical_crossentropy(ytrue:list, ypred:list):
  if len(ytrue) != len(ypred):
    raise Exception("y_true and y_pred must have the same length")
  
  answer = 0
  
  for i in range(len(ytrue)):
    
    if not 0 <= ypred[i] <= 1:
      raise ValueError(f"Predicted probabilities must be between 0 and 1 and not {ypred[i]}")
    
    if ytrue[i] == 1:
      answer -= math.log(ypred[i] + 1e-15)
      
  return answer

def Sparse_categorical_crossentropy(ytrue:int, ypred:list):
  if not all(0 <= p <= 1 for p in ypred):
    raise ValueError(f"Predicted probabilities must be between 0 or 1 and not {ypred}")

  if not isinstance(ytrue, int) or not (0 <= ytrue < len(ypred)):
    raise ValueError(f"True label must be an integer between 0 and {len(ypred) - 1} and not {ytrue}")

  probability = ypred[ytrue]
  
  answer = -math.log(probability + 1e-15)
      
  return answer
    
def Binary_crossentropy(ytrue:list, ypred:list):
  if len(ytrue) != len(ypred):
    raise Exception("y_true and y_pred must have the same length")
  
  answer = 0
  
  for y_true, y_pred in zip(ytrue, ypred):
    if not 0 <= y_true <= 1:
      raise Exception("target values must be 0 or 1")

    if not 0 <= y_pred <= 1:
      raise Exception("predicted values must be between 0 and 1")

    if y_true == 0:
      answer += (-math.log(1 - y_pred)) if y_pred < 1 else 100
      
    elif y_true == 1:
      answer += (-math.log(y_pred)) if y_pred > 0 else 100
      
    else:
      raise Exception(f"target values must be 0 or 1 and not {y_true}, you ment {y_pred}")
    
  return answer

def Hinge_loss(ytrue:list, ypred:list):
  if len(ytrue) != len(ypred):
    raise Exception("y_true and y_pred must have the same length")

  answer = 0
  
  for y_true, y_pred in zip(ytrue, ypred):
    answer += max(1 - y_true * y_pred, 0)
  
  return answer

def l1_loss(ytrue:list, ypred:list):
  if len(ytrue) != len(ypred):
    raise Exception("y_true and y_pred must have the same length")

  answer = 0
  
  for y_true, y_pred in zip(ytrue, ypred):
    answer += abs(y_true - y_pred)
  
  return answer
