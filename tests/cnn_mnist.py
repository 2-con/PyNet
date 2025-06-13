import sys
import os
# Get the directory containing 'pynet'
current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

from api import synapse as net

from tools.visual import image_display
from tools.scaler import argmax
from datasets.image import mnist
from tools.arraytools import generate_random_array
import matplotlib.pyplot as plt

train_images, train_labels, test_images, test_labels = mnist(one_hot=True, normalized=True).load()

  
model = net.Sequential( 
    
  net.Parallel(
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
  ),
  net.Parallel(
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
  ),
  net.Parallel(
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
    net.Convolution((3,3), 'elu'),
  ),
  net.Parallel(
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
    net.Meanpooling(2,2),
  ),
  net.Merge('total'),
  net.Flatten(),
  net.Dense(64, 'elu'),
  net.Dense(64, 'elu'),
  net.Dense(32, 'elu'),
  net.Dense(10, 'none'),
  net.Operation('softmax')
)

model.compile(
  optimizer='adam',
  loss='categorical crossentropy',
  metrics=['accuracy'],
  batchsize = 10,
  learning_rate=0.001,
  epochs=1,
  verbose=4,
  regularity=1
)

model.fit(
  train_images[:50],
  train_labels[:50],
)

# aesthetics

img_index = 67
image_display(train_images[img_index])
print("num | true | predicted")
count = 0
predicted_label = model.push(train_images[img_index])
for predicted, true in zip(predicted_label, train_labels[img_index]):
  print(f" {count}  | {true}    | {predicted}")
  count += 1
print()
print(f"{argmax(predicted_label) == argmax(train_labels[img_index])}    {argmax(train_labels[img_index])} | {argmax(predicted_label)}")

model.evaluate(test_images[:5000], test_labels[:5000])

plt.plot(range(len(model.error_logs)), model.error_logs)
plt.title("Model Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel(f" {model.loss} Loss")
plt.show()