import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vgg16 import VGG16
import torch
import torch.nn as nn

# setting
batch_size = 100
learning_rate = 0.0002
num_epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

# Data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

cifar10_train = datasets.CIFAR10(root='./Data/',train=True,transform=transform,target_transform=None,download=True)
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transform, target_transform=None, download=True)

train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(cifar10_test, batch_size=batch_size)

# Train
model = VGG16(base_dim=64).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_arr = []

for i in range(num_epoch): 
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        y_ = label.to(device)

        optimizer.zero_grad() #optimizer의 gradient를 0으로 설정
        output = model.forward(x)
        # _, output = torch.max(output, 1)
        #print(output.shape)
        #print(y_.shape)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()

    if i%10 ==0:
        print(f'epcoh {i} loss : ',loss)
        loss_arr.append(loss.cpu().detach().numpy()) #detach tensor를 gradient 연산에서 분리

torch.save(model.state_dict(), "./train_model/VGG16_100.pth")