import math
import pynet.core.loss as loss
import numpy as np

def Accuracy(y_true:list, y_pred:list):
  """
  Accuracy
  -----
  Returns the accuracy of the model
  -----
  Args
  -----
  - y_true (list) : the true values
  - y_pred (list) : the predicted values

  Returns
  -----
    (float) : the accuracy of the model
  """
  correct = 0

  for true, pred in zip(y_true, y_pred):
    if true == pred:
      correct += 1

  return correct / len(y_true)

def Precision(y_true:list, y_pred:list):
  """
  Precision
  -----
  Returns the precision of the model
  -----
  Args
  -----
  - y_true (list) : the true values
  - y_pred (list) : the predicted values

  Returns
  -----
    (float) : the precision of the model
  """
  true_positive = 0
  false_positive = 0

  for true, pred in zip(y_true, y_pred):
    if true == 1 and pred == 1:
      true_positive += 1

    elif true == 0 and pred == 1:
      false_positive += 1

  return true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0.0

def Recall(y_true:list, y_pred:list):
  """
  Recall
  -----
  Returns the recall of the model
  -----
  Args
  -----
  - y_true (list) : the true values
  - y_pred (list) : the predicted values

  Returns
  -----
    (float) : the recall of the model
  """
  true_positive = 0
  false_negative = 0

  for true, pred in zip(y_true, y_pred):
    if true == 1 and pred == 1:
      true_positive += 1

    elif true == 1 and pred == 0:
      false_negative += 1

  return true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0.0

def F1_score(y_true:list, y_pred:list):
  """
  F1 Score
  -----
  Returns the F1 score of the model
  -----
  Args
  -----
  - y_true (list) : the true values
  - y_pred (list) : the predicted values

  Returns
  -----
    (float) : the F1 score of the model
  """
  precision = Precision(y_true, y_pred)
  recall = Recall(y_true, y_pred)

  return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

def ROC_AUC(y_true:list, y_pred:list):
  """
  ROC AUC
  -----
  Returns the ROC AUC of the model
  -----
  Args
  -----
  - y_true (list) : the true values
  - y_pred (list) : the predicted values

  Returns
  -----
    (float) : the ROC AUC of the model
  """
  # Placeholder for ROC AUC calculation, as it requires more complex logic and data structures.
  return None

def Log_Loss(y_true:list, y_pred:list):
  """
  Log Loss
  -----
  Returns the log loss of the model
  -----
  Args
  -----
  - y_true (list) : the true values
  - y_pred (list) : the predicted values

  Returns
  -----
    (float) : the log loss of the model
  """
  return -sum([true * math.log(pred) + (1 - true) * math.log(1 - pred) for true, pred in zip(y_true, y_pred)]) / len(y_true)

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
  return math.sqrt(loss.Mean_squared_error(y_true, y_pred))

def R2_Score(y_true:list, y_pred:list):
  """
  R2 Score
  -----
  Returns the R2 score of the model
  -----
  Args
  -----
  - y_true (list) : the true values
  - y_pred (list) : the predicted values

  Returns
  -----
    (float) : the R2 score of the model
  """
  ss_total = sum([(true - np.mean(y_true))**2 for true in y_true])
  ss_residual = sum([(true - pred)**2 for true, pred in zip(y_true, y_pred)])

  return 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
