import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vgg16 import VGG16
import torch
import torch.nn as nn

#setting
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 100

# Data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)

test_loader = DataLoader(cifar10_test, batch_size=batch_size)

# Train
model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load('./train_model/VGG16_100.pth'))

# eval
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for i, [image,label] in enumerate(test_loader):
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)
        _, output_index = torch.max(output,1) # 여기서 1은 뭐지? #인덱스 찾는거임

        total += label.size(0)
        correct += (output_index==y).sum().float()
    
    print("Accuracy of Test Data: {}%".format(100*correct/total))