import torch
import torch.nn as nn
import torch.optim as optim
from data_loader.cifar10_loader import Cifar10Loader
from convnet.convnet import AlexNet


def test_accuracy(net: nn.Module, data_loader: Cifar10Loader) -> float:
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        for data in data_loader.test_loader:
            inputs, labels = data
            outputs = net(inputs)
            _, prediction = torch.max(input=outputs.data, dim=1)
            total_count += labels.size(0)
            correct_count += (prediction == labels).sum().item()

    return correct_count / total_count


def main():
    # hyper-parameters
    batch_size = 8
    learning_rate = 1e-3
    momentum = 0.9
    epochs = 2

    # non hyper-parameter configs
    loss_print_step = 1000
    model_save_path = 'model/alexnet-cifar10.pth'

    # CIFAR10 loader
    print("[DataLoader] Loading CIFAR10")
    data_loader = Cifar10Loader(batch_size=batch_size, path='./data')

    # Network
    net = AlexNet()

    # optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=learning_rate, momentum=momentum)

    # train
    for epoch in range(1, epochs + 1):
        running_loss = 0

        for i, data in enumerate(data_loader.train_loader, start=1):
            # load mini-batch
            inputs, labels = data

            # run model
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # train model
            loss.backward()
            optimizer.step()

            # print loss
            running_loss += loss.item()
            if i % loss_print_step == 0:
                torch.save(obj=net.state_dict(), f=model_save_path)
                print(f"[Epoch {epoch}, minibatch {i}] Loss: {running_loss / 2000}")
                running_loss = 0

        # test accuracy
        accuracy = test_accuracy(net=net, data_loader=data_loader)
        print(f"[Epoch {epoch}] Test accuracy: {accuracy}")


if __name__ == '__main__':
    main()
