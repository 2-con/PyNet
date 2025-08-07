import math
import numpy as np
from tools.scaler import argmax

def Accuracy(y_true: list, y_pred: list):
  """
  Accuracy
  -----
    Returns the accuracy of the model.
    Applicable for both binary and multi-class classification.
  -----
  Args
  -----
  - y_true (list) : The true class labels (e.g., [0, 1, 0] or ['cat', 'dog']).
  - y_pred (list) : The predicted class labels (must be hard predictions, e.g., [0, 1, 1] or ['cat', 'bird']).

  Returns
  -----
    (float) : The proportion of correctly classified samples. Returns 0.0 for empty input.
  """
  if not y_true:
    return 0.0
  
  correct = 0
  
  for true, pred in zip(y_true, y_pred):
    if argmax(true) == argmax(pred):
      correct += 1
      
  return (correct / len(y_true)) * 100

def Precision(y_true: list, y_pred: list):
  """
  Precision (for Binary Classification)
  -----
    Returns the precision of the model for the positive class (assumed to be 1).
    Precision = TP / (TP + FP)
    It measures the proportion of positive identifications that were actually correct.
  -----
  Args
  -----
  - y_true (list) : The true binary labels (0 or 1).
  - y_pred (list) : The predicted binary labels (hard predictions, 0 or 1).

  Returns
  -----
    (float) : The precision score. Returns 0.0 if (TP + FP) is zero (no positive predictions).
  """
  if not y_true: # Handle empty input case
    return 0.0
  
  true_positive = 0
  false_positive = 0

  for true, pred in zip(y_true, y_pred):
    if argmax(true) == 1 and argmax(pred) == 1: # True Positive
      true_positive += 1
      
    elif argmax(true) == 0 and argmax(pred) == 1: # False Positive
      false_positive += 1

  denominator = true_positive + false_positive
  return (true_positive / denominator) * 100 if denominator != 0 else 0.0

def Recall(y_true: list, y_pred: list):
  """
  Recall (for Binary Classification)
  -----
    Returns the recall (sensitivity or True Positive Rate) of the model for the positive class (assumed to be 1).
    Recall = TP / (TP + FN)
    It measures the proportion of actual positives that were identified correctly.
  -----
  Args
  -----
  - y_true (list) : The true binary labels (0 or 1).
  - y_pred (list) : The predicted binary labels (hard predictions, 0 or 1).

  Returns
  -----
    (float) : The recall score. Returns 0.0 if (TP + FN) is zero (no actual positives).
  """
  if not y_true: # Handle empty input case
    return 0.0
  true_positive = 0
  false_negative = 0

  for true, pred in zip(y_true, y_pred):
    if argmax(true) == 1 and argmax(pred) == 1: # True Positive
      true_positive += 1
    elif argmax(true) == 1 and argmax(pred) == 0: # False Negative
      false_negative += 1

  denominator = true_positive + false_negative
  return (true_positive / denominator) * 100 if denominator != 0 else 0.0

def F1_score(y_true: list, y_pred: list):
  """
  F1 Score (for Binary Classification)
  -----
    Returns the F1 score, which is the harmonic mean of Precision and Recall.
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    It's a balanced measure that considers both precision and recall.
  -----
  Args
  -----
  - y_true (list) : The true binary labels (0 or 1).
  - y_pred (list) : The predicted binary labels (hard predictions, 0 or 1).

  Returns
  -----
    (float) : The F1 score. Returns 0.0 if (Precision + Recall) is zero.
  """
  precision = Precision(y_true, y_pred)/100
  recall = Recall(y_true, y_pred)/100

  denominator = precision + recall
  return 2 * (precision * recall) / denominator if denominator != 0 else 0.0

def ROC_AUC(y_true: list, y_pred_prob: list):
  """
  ROC AUC (Area Under the Receiver Operating Characteristic Curve)
  -----
    Returns the ROC AUC score for binary classification.
    Requires predicted probabilities for the positive class.
    Measures the model's ability to distinguish between classes across various thresholds.
  -----
  Args
  -----
  - y_true (list) : The true binary labels (0 or 1).
  - y_pred_prob (list) : The predicted probabilities for the positive class (between 0 and 1).

  Returns
  -----
    (float) : The ROC AUC score. Returns 0.5 if all predictions are the same or not enough positives/negatives.
  """
  if not y_true or len(y_true) != len(y_pred_prob):
      raise ValueError("y_true and y_pred_prob must be non-empty lists of the same length.")
  
  # Ensure there are at least two distinct true classes for a meaningful AUC
  if len(set(y_true)) < 2: 
      return 0.5 # By convention, AUC is 0.5 for a single class (random guessing)

  # Pair predictions with true labels and sort by prediction probability in descending order
  paired_data = sorted(zip(y_pred_prob, y_true), key=lambda x: x[0], reverse=True)

  tprs = [0.0] # True Positive Rates (Recall)
  fprs = [0.0] # False Positive Rates

  # Total counts for actual positives and negatives
  n_pos = y_true.count(1)
  n_neg = y_true.count(0)

  # Handle cases with no positive or no negative samples in y_true
  if n_pos == 0 or n_neg == 0:
      return 0.5 # Cannot calculate AUC for a single class

  tp = 0 # Current true positives count
  fp = 0 # Current false positives count
  
  # Iterate through unique thresholds (predicted probabilities)
  # This correctly handles ties in predicted probabilities.
  # We add points for all unique prediction values.
  # The ROC curve starts at (0,0) and ends at (1,1).
  
  # Group by unique probabilities for efficient iteration
  unique_prob_groups = []
  current_group_true = 0
  current_group_false = 0
  
  if paired_data:
      current_prob = paired_data[0][0]
      for prob, label in paired_data:
          if prob != current_prob:
              unique_prob_groups.append((current_prob, current_group_true, current_group_false))
              current_prob = prob
              current_group_true = 0
              current_group_false = 0
          if label == 1:
              current_group_true += 1
          else:
              current_group_false += 1
      unique_prob_groups.append((current_prob, current_group_true, current_group_false)) # Add last group

  # Generate ROC points
  for prob_val, group_true, group_false in unique_prob_groups:
      tp += group_true
      fp += group_false
      
      current_tpr = tp / n_pos
      current_fpr = fp / n_neg
      
      # Add only if the point is new to avoid redundant points (especially with ties)
      if (current_fpr, current_tpr) != (fprs[-1], tprs[-1]):
          fprs.append(current_fpr)
          tprs.append(current_tpr)
          
  # Ensure the curve starts at (0,0) and ends at (1,1) if not already there
  if (fprs[0], tprs[0]) != (0.0, 0.0):
      fprs.insert(0, 0.0)
      tprs.insert(0, 0.0)
  if (fprs[-1], tprs[-1]) != (1.0, 1.0):
      fprs.append(1.0)
      tprs.append(1.0)
      
  auc = 0.0
  # Calculate area using the trapezoidal rule
  for i in range(len(fprs) - 1):
      # Area of trapezoid = 0.5 * (height1 + height2) * width
      auc += (fprs[i+1] - fprs[i]) * (tprs[i+1] + tprs[i]) / 2.0
      
  return auc

def R2_Score(y_true: list, y_pred: list):
  """
  R2 Score (Coefficient of Determination)
  -----
    Returns the R2 score for regression tasks.
    It indicates the proportion of the variance in the dependent variable
    that is predictable from the independent variables.
    R2 = 1 - (SS_res / SS_tot)
  -----
  Args
  -----
  - y_true (list) : The true numerical values.
  - y_pred (list) : The predicted numerical values.

  Returns
  -----
    (float) : The R2 score. Returns 0.0 if total sum of squares is zero (i.e., all true values are identical).
  """
  if not y_true or len(y_true) != len(y_pred): # Handle empty or mismatched input
    return 0.0
    
  mean_y_true = np.mean(y_true) 

  ss_total = sum([(true - mean_y_true)**2 for true in y_true])
  ss_residual = sum([(true - pred)**2 for true, pred in zip(y_true, y_pred)])

  if ss_total == 0:
    # If ss_total is 0, it means all y_true values are identical.
    # In this case, R2 is typically 0 if ss_residual is also 0, or undefined.
    return 0.0 if ss_residual == 0 else -1e+10 # Or 0.0 for consistent pred; -inf if wrong pred
  
  return 1 - (ss_residual / ss_total)
