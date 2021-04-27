import torch.optim as optim
import torch.nn as nn
from alexnet import AlexNet
import torch.utils.data


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = AlexNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=1e-3, momentum=0.9)


for epoch in range(2):
    running_loss = 0
    for i, data in enumerate(cifar10_train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"epoch: {epoch}, minibatch: {i}, loss: {running_loss / 2000}")
            running_loss = 0

print('finished training')
torch.save(obj=net.state_dict(), f='../model/cifar_net.pth')
