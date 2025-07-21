
def OneHotEncode(length:int, *args:int,):
  """
  Manually create a single one-hot encoded vector.
  """
  answer = [0 for _ in range(length)]

  for x in args:
    answer[x] = 1

  return answer

def OneHotEncoder(data):
  """
  One Hot Encoder
  -----
    Automatically encode every vector in a list based on its unique elements.
  -----
  Args
  -----
  - data (list): List of elements to be one-hot encoded.
  """
  unique_elements = list(set(data))
  answer = []
  
  for element in data:
    encoded_vector = [1 if elem == element else 0 for elem in unique_elements]
    answer.append(encoded_vector)
    
  return answer

def OrdinalEncoder(ranking, data):
  """
  Ordinal Encoder
  -----
    Automatically encode every vector in a list based on its unique elements.
  -----
  Args
  -----
  - ranking (list): List of unique elements in the order of their rank, starting from 0.
  - data (list): List of elements to be ordinal encoded.
  """
  ranks = {value: index for index, value in enumerate(ranking)}
  answer = []
  
  for element in data:
    if element in ranks:
      answer.append(ranks[element])
    else:
      raise ValueError(f"'{element}' not found in ranking list.")
    
  return answer
