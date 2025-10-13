import torch

PATH = "./cat_dog.pth"

net = torch.load(PATH, weights_only=False)
net.eval()
