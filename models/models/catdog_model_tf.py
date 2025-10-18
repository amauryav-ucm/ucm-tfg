import tensorflow as tf
from tensorflow.keras import layers, models


class Classes:
    def __init__(self):
        self.classes = ("Cat", "Dog")

    def __getitem__(self, index):
        return self.classes[index]

    def __len__(self):
        return len(self.classes)


class Transform:
    def __init__(self):
        self.resize = layers.Resizing(60, 60)
        self.normalize = layers.Rescaling(1.0 / 255)
        self.standardize = layers.Normalization(
            mean=[0.5, 0.5, 0.5], variance=[0.25, 0.25, 0.25]
        )

    def __call__(self, img):
        img = self.resize(img)
        img = self.normalize(img)
        return self.standardize(img)


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.conv1 = layers.Conv2D(6, (5, 5), activation="relu")
        self.pool = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(16, (5, 5), activation="relu")
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(120, activation="relu")
        self.fc2 = layers.Dense(84, activation="relu")
        self.fc3 = layers.Dense(2)  # logits

    def call(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
