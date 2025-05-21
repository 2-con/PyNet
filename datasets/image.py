import json

def OneHotEncoder(length: int, *args: int):
  """
  Binary categorization
  """
  answer = [0 for _ in range(length)]
  for x in args:
    answer[x] = 1
  return answer

if __name__ == "__main__":
  from keras.datasets import mnist
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  # Convert NumPy arrays to Python lists
  train_images_list             = X_train.tolist()
  train_images_list_normalized  = (X_train/255).tolist()
  train_labels_list             = y_train.tolist()
  onehot_train_labels_list      = [OneHotEncoder(10, x) for x in y_train.tolist()]

  test_images_list              = X_test.tolist()
  test_images_list_normalized   = (X_test/255).tolist()
  test_labels_list              = y_test.tolist()
  onehot_test_labels_list       = [OneHotEncoder(10, x) for x in y_test.tolist()]

  mnist_data = {
    'train_images'            : train_images_list,
    'normalized_train_images' : train_images_list_normalized,
    'train_labels'            : train_labels_list,
    'onehot_train_labels'     : onehot_train_labels_list,
    
    'test_images'             : test_images_list,
    'normalized_test_images'  : test_images_list_normalized,
    'test_labels'             : test_labels_list,
    'onehot_test_labels'      : onehot_test_labels_list
  }

  # Save the data to a JSON file
  with open('mnist.json', 'w') as f:
    json.dump(mnist_data, f)

  print("MNIST data saved to mnist.json as Python lists.")

class mnist:
  def __init__(self, one_hot=False, normalized=False):
    self.one_hot = one_hot
    self.normalized = normalized
    with open(r'C:\Users\User\OneDrive\Desktop\homework\Coding\Python\pynet\datasets\mnist.json', 'r') as f:
      self.data = json.load(f)
  
  def load(self):
    """
    load
    -----
      Returns the MNIST dataset as a tuple of 4 lists. If one_hot is True, the labels are one-hot encoded.
    -----
      Returns:
      - train_images: a list of 60000 images of size 28x28
      - train_labels: a list of 60000 labels, each is a one-hot encoded vector if one_hot is True
      - test_images: a list of 10000 images of size 28x28
      - test_labels: a list of 10000 labels, each is a one-hot encoded vector if one_hot is True
    """
  
    return (
      self.data['normalized_train_images'] if self.normalized else self.data['train_images'],
      self.data['onehot_train_labels'] if self.one_hot else self.data['train_labels'],
      
      self.data['normalized_test_images'] if self.normalized else self.data['test_images'],
      self.data['onehot_test_labels'] if self.one_hot else self.data['test_labels']
    )