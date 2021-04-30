import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_loader.cifar10_loader import Cifar10Loader
from distributed_library.mpi import all_reduce
from alexnet.alexnet import AlexNet


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
    # data parallel configs
    npus_count = 4

    # hyper-parameters
    batch_size = 8
    learning_rate = 1e-3
    momentum = 0.9
    epochs = 2

    # non hyper-parameter configs
    loss_print_step = 1000
    model_save_path = 'model/alexnet-cifar10-distributed.pth'

    # CIFAR10 loader
    print("[DataLoader] Loading CIFAR10")
    data_loader = Cifar10Loader(batch_size=batch_size, path='./data')

    # Initialize networks
    net = AlexNet()
    partial_nets = [AlexNet() for _ in range(npus_count)]

    # Synchronize initialized values
    for npu in range(npus_count):
        partial_nets[npu].load_state_dict(state_dict=net.state_dict())

    # optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=learning_rate, momentum=momentum)

    # train
    for epoch in range(epochs):
        running_loss = 0

        for minibatch_index, data in enumerate(data_loader.train_loader, start=1):
            # load mini-batch
            inputs, labels = data

            # split into micro-batches
            microbatch_size = inputs.shape[0] // npus_count
            inputs_split = torch.split(inputs, microbatch_size, dim=0)
            labels_split = torch.split(labels, microbatch_size, dim=0)

            # run microbatches
            outputs = list()
            losses = list()

            optimizer.zero_grad()

            for npu in range(npus_count):
                # reset partial grad
                partial_nets[npu].zero_grad()

                # run forward pass
                partial_output = partial_nets[npu](inputs_split[npu])
                partial_loss = criterion(partial_output, labels_split[npu])

                outputs.append(partial_output)
                losses.append(partial_loss)

                # run backprop (partial)
                partial_loss.backward()

            # create gradient messages
            gradient = dict()
            for name, _ in net.named_parameters():
                gradient[name] = dict()
                for npu in range(npus_count):
                    gradient[name][npu] = dict()

            for npu in range(npus_count):
                for name, param in partial_nets[npu].named_parameters():
                    linear_param = param.grad.clone().reshape(-1)

                    padding_size = npus_count - (linear_param.shape[0] % npus_count)
                    linear_param = F.pad(input=linear_param, pad=(0, padding_size), value=0)
                    message_size = linear_param.shape[0] // npus_count

                    param_split = torch.split(linear_param, message_size, dim=0)

                    for message_id in range(npus_count):
                        gradient[name][npu][message_id] = param_split[message_id]

            # do all-reduce
            for name, _ in net.named_parameters():
                all_reduce(nodes_count=npus_count, message=gradient[name])

            # apply gradient
            for name, params in net.named_parameters():
                # concate message
                tensors_to_concat = list()
                for npu in range(npus_count):
                    tensors_to_concat.append(gradient[name][0][npu])
                new_params = torch.cat(tensors_to_concat, dim=0)

                # reshape
                param_size = params.view(-1).shape[0]
                new_params = new_params[:param_size]
                new_params = new_params.reshape(params.shape)

                # update new params
                params.grad = new_params

            # train model
            optimizer.step()

            # Synchronize initialized values
            for npu in range(npus_count):
                partial_nets[npu].load_state_dict(state_dict=net.state_dict())

            # print loss
            running_loss += sum(map(lambda x: x.item(), losses))
            if minibatch_index % loss_print_step == 0:
                torch.save(obj=net.state_dict(), f=model_save_path)
                print(f"[Epoch {epoch}, minibatch {minibatch_index}] Loss: {running_loss / loss_print_step}")
                running_loss = 0

        # test accuracy
        accuracy = test_accuracy(net=net, data_loader=data_loader)
        print(f"[Epoch {epoch}] Test accuracy: {accuracy}")


if __name__ == '__main__':
    main()
