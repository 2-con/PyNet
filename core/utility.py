

def train_test_split(data, test_size=0.2):
  """
  Train/Test Split
  -----
    Splits data into training and testing sets
  -----
  Args
  -----
  - test_size (float) : controls how much of the original dataset will go towards testing

  Returns
  -----
    (Testing data, Training data)
  """
  train_size = 1 - test_size
  return data[:int(train_size*len(data))], data[int(train_size*len(data)):]
