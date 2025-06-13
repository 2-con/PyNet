

def split(data, second_size=0.2):
  """
  Split
  -----
    Splits a list into two lists of configurable lengths, 
    used for splitting data into training and testing sets
  -----
  Args
  -----
  - second_size (float) : controls how much of the original list will go towards the second list

  Returns
  -----
    (1st half, second half)
  """
  first_size = int((1 - second_size) * len(data))
  return data[:first_size], data[first_size:]

