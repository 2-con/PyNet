import sys
import os
# Get the directory containing 'pynet'
current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

import api.synapse as tf

from tools.visual import image_display
from tools.scaler import argmax
from datasets.image import mnist
import matplotlib.pyplot as plt

train_images, train_labels, test_images, test_labels = mnist(one_hot=True, normalized=True).load()

# model architecture
model = tf.Sequential()
model.add(tf.Flatten())
model.add(tf.Dense(128, activation='elu'))
model.add(tf.Dense(32, activation='elu'))
model.add(tf.Dense(32, activation='elu'))
model.add(tf.Dense(10, activation='elu'))
model.add(tf.Operation(operation='softmax'))

model.compile(
  optimizer='adam',
  loss='categorical crossentropy',
  metrics=['accuracy'],
  batchsize = 1,
  learning_rate=0.1,
  epochs=100
)

model.fit(
  train_images[:10],
  train_labels[:10],
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