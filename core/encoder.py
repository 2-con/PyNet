
def OneHotEncoder(length:int, *args:int,):
  """
  Binary categorization
  """
  answer = [0 for _ in range(length)]

  for x in args:
    answer[x] = 1

  return answer
