"""
Encryption
-----
Do not use this for commercial purposes, these are just demos

"""

def encrypt(word, key, format):
  """
  Do not use this for commercial purposes.
  -----
  This is merely a simple encryption function to convert plaintext to encrypted text. it is unsecure by modern, industry standards
  """
  # 92 characters long including spaces, start counting from 0, no specials
  chars = list(str(format))

  numword = []
  numkey = []
  answer = []
  place = []
  ans = ""
  step = 0
  found = False
  done = False
  dupechar = []
  place1 = []
  place2 = []

  # chack if there are any duplicate characters

  for i in range(len(format)):
    for i in range(len(chars)):
      if done==False:
        place.append(i)
      else:
        if format[step] == chars[i] and step != i:
          dupechar.append(chars[i])
          place1.append(step)
          place2.append(i)
    step += 1
    done = True
  step = 0

  # in case there are duplicates, show the user

  if len(dupechar) != 0:
    for i in range(len(dupechar)//2):
      print(f"duplicate detected = {dupechar[i]}, charachter {place1[i]} and {place2[i]} matched")
    exit()

  # encryption process ##########################################################

  # numbers word

  for i in word:
    while found == False:
      if i == chars[step]:
        numword.append(place[step])
        found = True
        step = 0
      else:
        step+=1
    found = False

  step = 0
  found = False

  # numbers the key

  for i in key:
    # checks if the key corresponds to a character
    while found == False:
      if i == chars[step]:
        numkey.append(place[step])
        found = True
        step = 0
      else:
        step+=1
    found = False

  step = 0

  # padding the key

  while len(numword) > len(numkey):
    numkey.append(numkey[step])
    step += 1

  step = 0

  # encode the message by key and length of key

  for i in numword:
    answer.append(i+(numkey[step]+len(key)))
    step += 1

  step = 0

  # final answer

  for i in answer:
    # checks if [numbered] overflows the list of characters
    if i<=(place[-1]):
      ans = ans + chars[i]
    else:
      ans = ans + chars[i%(place[-1]+1)]

  # debugging

  #print(f"from encrypt, numword {numword}")
  #print(f"from encrypt, answer  {answer}")

  return ans

def decrypt(word, key, format):
  """
  Do not use this for commercial purposes.
  -----
  This is merely a simple decryption function to convert encrypted text to plaintext. it is unsecure by modern, industry standards
  """
  
  chars = list(str(format))

  numword = []
  numkey = []
  answer = []
  place = []
  ans = ""
  step = 0
  found = False
  done = False
  dupechar = []
  place1 = []
  place2 = []

  # pre-processing
  # chack if there are any duplicate characters

  for i in range(len(format)):
    for i in range(len(chars)):
      if done==False:
        place.append(i)
      else:
        if format[step] == chars[i] and step != i:
          dupechar.append(chars[i])
          place1.append(step)
          place2.append(i)
    step += 1
    done = True
  step = 0

  # in case there are duplicates, show the user

  if len(dupechar) != 0:
    for i in range(len(dupechar)//2):
      print(f"duplicate detected = {dupechar[i]}, charachter {place1[i]} and {place2[i]} matched")
    exit()

  # actual decryption ##########################################################

  # numbers the word

  for i in word:
    # checks if char corresponds to a num
    while found == False:
      if i == chars[step]:#
        numword.append(place[step])
        found = True
        step = 0
      else:
        step+=1
    found = False

  step = 0
  found = False

  # numbers the key

  for i in key:
    # checks if char corresponds to a num
    while found == False:
      if i == chars[step]:
        numkey.append(place[step])
        found = True
        step = 0
      else:
        step+=1
    found = False

  step = 0

  # padding the key

  while len(numword) > len(numkey):
    numkey.append(numkey[step])
    step += 1

  step = 0

  # decode message with key

  for i in numword:
    answer.append((i-(numkey[step]+len(key)))%(place[-1]+1))
    step += 1

  step = 0

  # final answer & debugging

  #print(f"from decrypt, answer  {answer}")

  for i in answer:
    ans = ans + chars[i]

  return ans

def hash(word, hashtype, format=" abcdefghijklmnopqrstuvwxyzZYXWVUTSRQPONMLKJIHGFEDCBA0123456789!?@#$%&*~-=_+/[]{}<>;:|,.`'\""):
  """
  Do not use this for commercial purposes.
  -----
  This is merely a simple hashing function to convert text to a hash. it is unsecure by modern, industry standards
  """
  
  # premade hashtypes
  
  HASHLENGTH = 128
  COMPLEXITY = 3
  
  if hashtype == "SHA256":
    hashtype = "12a3b4c5d6e7f8g90"
    HASHLENGTH = 256
  elif hashtype == "AOS256":
    hashtype = "abcdefghijklmnopqrstuvwxyzZYXWVUTSRQPONMLKJIHGFEDCBA"
    HASHLENGTH = 256
  elif hashtype == "AOS256++":
    hashtype = "abcdefghijklmnopqrstuvwxyzZYXWVUTSRQPONMLKJIHGFEDCBA0123456789!?@#$%&*~-=_+/[]{}<>;:|,.`'\""
    HASHLENGTH = 256
    COMPLEXITY = 2
  elif hashtype == "AOS3":
    hashtype = "abcdefghijklmnopqrstuvwxyzZYXWVUTSRQPONMLKJIHGFEDCBA0123456789"
    COMPLEXITY = 5
  elif hashtype == "AOS2":
    hashtype = "abcdefghijklmnopqrstuvwxyz0123456789"
    COMPLEXITY = 5
    HASHLENGTH = 64
  else:
    hashtype = "abcdefghijklmnopqrstuvwxyzZYXWVUTSRQPONMLKJIHGFEDCBA0123456789!?@#$%&*~-=_+/[]{}<>;:|,.`'\""
  
  chars = list(str(format))
  
  for i in range(len(word)):
    place = []
    numword = []
    answer = []
    ans = ""
    step = 0
    hashloop = 0
    final = 0
    found = False
    done = False
    dupechar = []
    place1 = []
    place2 = []

    # setting up the process
    # chack if there are any duplicate characters in the encoding map

    for i in range(len(format)):
      for i in range(len(chars)):
        if done==False:
          place.append(i)
        else:
          if format[step] == chars[i] and step != i:
            dupechar.append(chars[i])
            place1.append(step)
            place2.append(i)
      step += 1
      done = True

    step = 0

    # in case there are duplicates, show the user

    if len(dupechar) != 0:
      for i in range(len(dupechar)//2):
        print(f"duplicate detected = {dupechar[i]}, charachter {place1[i]} and {place2[i]} matched")
      exit()

    # HASHING process ##########################################################

    # pads the word if its shorter than required

    while len(word) < HASHLENGTH:
      word = word + chars[step%len(hashtype)]
      step += 1

    step = 0

    # numbers the word

    for i in word:
      found = False
      while found == False:
        if step > place[-1]:
          while step > place[-1]:
            step=-place[-1]

        if i == chars[step]:
          numword.append(place[step])
          step = 0
          found = True

        else:
          step+=1

    step = 0

    # psuedo-hashing the list

    for i in range(COMPLEXITY):

      # temporarly relocate numword so that i can do funny stuff to it
      temporary = numword
      numword = []

      # goes through all the elements in numbered, adding up all the previous numbers to that point for all elements
      for i in range(len(temporary)-1):
        # sums up everything until that point
        for x in temporary[:i]:
          step += x
        # engrave the result in numword
        numword.append(temporary[i] + (step))
        step = 0

      step = 0
      numword[::-1]

    # restricts the value of elements in numword

    temporary = numword
    numword = []

    for i in temporary:
      if i > len(hashtype):
        i %= len(hashtype)
      numword.append(i)

    step = 0

    # multiply charachter by its placement multiplied by length

    for i in numword:
      step += 1
      if i*(step*len(word)) <= len(hashtype):
        answer.append(hashtype[i*(step*len(word))])
      else:
        answer.append(hashtype[(i*(step*len(word)))%len(hashtype)])

    step = 0

    # convert list to string

    for i in answer:
      ans = ans + i

    # final proccessing

    ans = ans.replace(" ","")

    if len(ans) > HASHLENGTH:
      pass
    elif len(ans) < HASHLENGTH:
      while len(ans) < HASHLENGTH:
        ans += ans[step]
        step += 1

    step = 0

    word=ans[::-1]

  return ans
