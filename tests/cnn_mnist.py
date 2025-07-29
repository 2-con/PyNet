import sys
import os
# Get the directory containing 'pynet'
current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

from api import netcore as net

from datasets.image import mnist
import matplotlib.pyplot as plt
from tools.arraytools import generate_random_array as mock_mnist

train_images = mock_mnist(28, 28, 100)
train_labels = mock_mnist(10,     100)
test_images  = mock_mnist(28, 28, 100)
test_labels  = mock_mnist(10,     100)

  
model = net.Sequential( 
    
  # net.Convolution((4,4), 32, 'relu'),
  # net.Convolution((3,3), 8, 'relu'),
  net.Convolution((3,3), 8, 'relu'),
  net.Flatten(),
  net.Dense(128, 'relu'),
  net.Dense(32, 'relu'),
  net.Dense(10, 'none'),
  net.Operation('softmax')
)

model.compile(
  optimizer='adam',
  loss='mean squared error',
  metrics=['accuracy'],
  batchsize = 1,
  learning_rate=0.1,
  epochs=250,
  verbose=2,
  logging=1,
  validation_split=0.25,
  optimize=True,
)

model.fit(
  train_images,
  train_labels,
)

# aesthetics

# img_index = 67
# image_display(train_images[img_index])
# print("num | true | predicted")
# count = 0
# predicted_label = model.push(train_images[img_index])
# for predicted, true in zip(predicted_label, train_labels[img_index]):
#   print(f" {count}  | {true}    | {predicted}")
#   count += 1
# print()
# print(f"{argmax(predicted_label) == argmax(train_labels[img_index])}    {argmax(train_labels[img_index])} | {argmax(predicted_label)}")

model.evaluate(test_images[:5000], test_labels[:5000])

plt.plot(range(len(model.error_logs))           , model.error_logs           , color='red')
plt.plot(range(len(model.validation_error_logs)), model.validation_error_logs, color='blue')
plt.show()