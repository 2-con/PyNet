import sys
import os
# Get the directory containing 'pynet'
current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

import api.synapse as net

from tools.visual import image_display
from tools.scaler import argmax
from datasets.image import mnist
import matplotlib.pyplot as plt

train_images, train_labels, test_images, test_labels = mnist(one_hot=True, normalized=True).load()

model = net.Sequential( 
                        
  net.Flatten(),
  net.Dense(128, 'elu'),
  net.Dense(32, 'elu'),
  net.Dense(32, 'elu'),
  net.Dense(10, 'elu'),
  net.Operation('softmax')
)

model.compile(
  optimizer='adam',
  loss='categorical crossentropy',
  metrics=['accuracy'],
  batchsize = 1,
  learning_rate=0.002,
  epochs=100,
)

model.fit(
  train_images[:50],
  train_labels[:50],
  verbose=4,
  regularity=1
)

img_index = 19
image_display(train_images[img_index])
print("num | true | predicted")
count = 0
predicted_label = model.push(train_images[img_index])
for predicted, true in zip(predicted_label, train_labels[img_index]):
  print(f" {count}  | {true}    | {predicted}")
  count += 1
print()
print(f"{argmax(predicted_label) == argmax(train_labels[img_index])}    {argmax(train_labels[img_index])} | {argmax(predicted_label)}")

model.evaluate(test_images[:100], test_labels[:100])


plt.plot(range(len(model.error_logs)), model.error_logs)
plt.title("Model Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel(f" {model.loss} Loss")
plt.show()