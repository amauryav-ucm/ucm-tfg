import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models


class Classes:
    def __init__(self):
        self.classes = ("Cat", "Dog")

    def __getitem__(self, index):
        return self.classes[index]

    def __len__(self):
        return len(self.classes)


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.resize = layers.Resizing(60, 60)
        self.normalize = layers.Rescaling(1.0 / 255)
        self.standardize = layers.Normalization(
            mean=[0.5, 0.5, 0.5], variance=[0.25, 0.25, 0.25]
        )
        self.conv1 = layers.Conv2D(6, (5, 5), activation="relu")
        self.pool = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(16, (5, 5), activation="relu")
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation="relu")
        self.fc2 = layers.Dense(84, activation="relu")
        self.fc3 = layers.Dense(2)  # logits

    def call(self, x):
        x = self.resize(x)
        x = self.normalize(x)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def predict_image(model, path):
    classes = Classes()
    img = Image.open(path)
    processed_image = tf.keras.utils.img_to_array(img)
    processed_image = tf.expand_dims(processed_image, 0)

    output = model.predict(processed_image)
    probs = tf.nn.softmax(output[0])
    predicted = np.argmax(probs)

    plt.imshow(img)
    plt.xlabel(f"Predicted: {classes[predicted]} ({100*float(probs[predicted]):.2f}%)")
