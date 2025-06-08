import sys
import os
# Get the directory containing 'pynet'
current_script_dir = os.path.dirname(__file__)
pynet_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(pynet_root_dir)

from api import synapse as tf

from tools.visual import image_display
from tools.scaler import argmax
from datasets.image import mnist
from tools.arraytools import generate_random_array
import matplotlib.pyplot as plt

train_images, train_labels, test_images, test_labels = mnist(one_hot=True, normalized=True).load()

# model architecture
model = tf.Sequential()
model.add(tf.Convolution(kernel=generate_random_array(2,2, min=-1, max=1), activation='elu', bias=True, learnable=True))
model.add(tf.Maxpooling(2,2))
model.add(tf.Convolution(kernel=generate_random_array(2,2, min=-1, max=1), activation='elu', bias=True, learnable=True))
model.add(tf.Maxpooling(2,2))
model.add(tf.Convolution(kernel=generate_random_array(2,2, min=-1, max=1), activation='elu', bias=True, learnable=True))
model.add(tf.Flatten())
model.add(tf.Dense(16, activation='elu'))
model.add(tf.Dense(10, activation='elu'))
model.add(tf.Operation(operation='softmax'))

model.compile(
  optimizer='adam',
  loss='categorical crossentropy',
  metrics=['accuracy'],
  batchsize = 1,
  learning_rate=0.011,
  epochs=250,
)

model.fit(
  train_images[:1000],
  train_labels[:1000],
  verbose=4,
  regularity=1
)

# aesthetics

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

model.evaluate(test_images[:1000], test_labels[:1000])

plt.plot(range(len(model.error_logs)), model.error_logs)
plt.title("Model Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel(f" {model.loss} Loss")
plt.show()