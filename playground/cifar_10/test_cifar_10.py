""" Tests the ResNet model on Cifar10 with AdamW """

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Meta
from kish.runners import train_network, test_network
from kish.classes import Network
from kish.graphing import report_network_loss, save_network_loss_graph
from kish.utils import save_network_output, save_network_test_results

# Model Specifications
from modules.ResNet import ResNet
from modules.Optimizer import SGDNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Define the transformation for the dataset
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transformtest = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transformtest
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )
    resnet = ResNet(20, 10)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = SGDNorm(resnet.parameters(), lr=0.1, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [81, 122], 0.1)

    input = Network.TrainingInput(
        trainloader=trainloader,
        epochs=4,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    output = train_network(
        resnet,
        input,
    )

    save_network_loss_graph(
        path="tests/cifar_10/graphs",
        input=output,
    )

    save_network_output(
        path="tests/cifar_10/outputs",
        input=output,
    )

    output = test_network(
        network=resnet,
        input=Network.TestingInput(
            testloader=testloader,
            device=device,
        ),
    )

    save_network_test_results(
        path="tests/cifar_10/results", input=output, name="CustomName"
    )
