import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


class Cifar10Loader:
    def __init__(self, batch_size: int, path: str = '../data', ):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        train_set = torchvision.datasets.CIFAR10(root=path,
                                                 train=True,
                                                 download=True,
                                                 transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=path,
                                                train=False,
                                                download=True,
                                                transform=transform)

        self.train_loader = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=0)
        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=0)
