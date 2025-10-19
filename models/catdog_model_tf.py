import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow.keras import layers, models, Input


class Classes:
    def __init__(self):
        self.classes = ("Cat", "Dog")

    def __getitem__(self, index):
        return self.classes[index]

    def __len__(self):
        return len(self.classes)


class Transform:

    def __call__(self, pic):
        pic = tf.image.resize(pic, (60, 60))
        pic = pic / 255
        return pic


def create_model():
    model = keras.Sequential(
        name="catdog_model",
        layers=[
            layers.InputLayer(input_shape=(60, 60, 3)),
            layers.Conv2D(6, 5, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 5, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(120, activation="relu"),
            layers.Dense(84, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ],
    )
    return model


def predict_image(model, path):
    classes = Classes()
    img = Image.open(path)
    processed_image = Transform()(img)
    processed_image = tf.expand_dims(processed_image, 0)

    output = model.predict(processed_image)
    print(output)
    probs = tf.nn.softmax(output)
    print(probs)
    predicted = np.argmax(probs)

    plt.imshow(img)
    plt.xlabel(f"Predicted: {classes[predicted]} ({100*float(probs[predicted]):.2f}%)")
