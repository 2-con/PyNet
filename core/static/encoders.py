
def OneHotEncode(length:int, *args:int,) -> list:
  """
  Manually create a single one-hot encoded vector.
  """
  answer = [0 for _ in range(length)]

  for x in args:
    answer[x] = 1

  return answer

def OneHotEncoder(data:list) -> list:
  """
  One Hot Encoder
  -----
    Automatically encode every vector in a list based on its unique elements.
  -----
  Args
  -----
  - data (list) : List of elements to be one-hot encoded.
  """
  unique_elements = list(set(data))
  answer = []
  
  for element in data:
    encoded_vector = [1 if elem == element else 0 for elem in unique_elements]
    answer.append(encoded_vector)
    
  return answer

def OrdinalEncoder(ranking:list, data:list) -> list:
  """
  Ordinal Encoder
  -----
    Automatically encode every vector in a list based on its unique elements.
  -----
  Args
  -----
  - ranking (list) : List of unique elements in the order of their rank, starting from 0.
  - data (list) : List of elements to be ordinal encoded.
  """
  ranks = {value: index for index, value in enumerate(ranking)}
  answer = []
  
  for element in data:
    if element in ranks:
      answer.append(ranks[element])
    else:
      raise ValueError(f"'{element}' not found in ranking list.")
    
  return answer

def Binarizer(threshold:float, data) -> list:
  """
  Binarizer
  -----
    Convert every numerical element in a list to either a 0 or a 1 depending of said element exceeds a certain threshold.
  -----
  Args
  -----
  - threshold (float or int) : The threshold value to compare against.
  - data (list) : List of numerical elements to be binarized.
  """
  return [1 if x > threshold else 0 for x in data]

def BinaryEncoder(data:list) -> list:
  """
  Binary Encoder
  -----
    Convert every numerical element in a list to binary where each number represent a scaler.
    simmilar to OneHotEncoder, it will return the data as a vector.
  -----
  Args
  -----
  - data (list) : List of numerical elements to be binarized. every element present must be an integer
  """
  
  return [bin(x)[2:].split() for x in data]

