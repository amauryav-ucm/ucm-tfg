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


@keras.saving.register_keras_serializable(package="Custom")
class Model(keras.Sequential):
    def __init__(self, build_model=True, **kwargs):
        super().__init__(**kwargs)
        if build_model:
            self.add(layers.InputLayer(input_shape=(60, 60, 3), name="catdog_input"))
            self.add(layers.Conv2D(6, 5, activation="relu"))
            self.add(layers.MaxPooling2D())
            self.add(layers.Conv2D(16, 5, activation="relu"))
            self.add(layers.MaxPooling2D())
            self.add(layers.Flatten())
            self.add(layers.Dense(120, activation="relu"))
            self.add(layers.Dense(84, activation="relu"))
            self.add(layers.Dense(2))

    @classmethod
    def from_config(cls, config):
        return cls(build_model=False, **config)


def predict_image(model, path):
    classes = Classes()
    img = Image.open(path)
    processed_image = Transform()(img)
    processed_image = tf.expand_dims(processed_image, 0)

    output = model.predict(processed_image)
    probs = tf.nn.softmax(output[0])
    predicted = np.argmax(probs)

    plt.imshow(img)
    plt.xlabel(f"Predicted: {classes[predicted]} ({100*float(probs[predicted]):.2f}%)")
