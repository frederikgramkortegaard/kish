""" Tests the ResNet model on Cifar10 with AdamW """

import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Meta
from kish.runners import train_network, test_network
from kish.classes import Network
from kish.graphing import save_network_loss_graph
from kish.utils import save_network_output, save_network_test_results, get_unique_id

# Model Specification
from model import ResNet

if __name__ == "__main__":
    # Define the transformation for the dataset
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomErasing(),
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

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    input = Network.TrainingInput(
        trainloader=trainloader,
        epochs=4,
        device=device,
        criterion=criterion,
        optimizer=None,
        scheduler=None,
    )

    # Setup a list of optimizers which we want to test, their parameters are formatted as kwargs and a custom name is given

    Optimizers = [
        (
            torch.optim.AdamW,
            {
                "lr": 0.003,
                "weight_decay": 1e-4,
            },
            "adamw",
        ),
        (
            torch.optim.SGD,
            {
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
            },
            "sgd",
        ),
        (
            torch.optim.SGD,
            {
                "lr": 0.1,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "nesterov": True,
            },
            "sgd-nesterov",
        ),
    ]

    for optim, params, name in Optimizers:
        resnet = ResNet(56, 10)

        input.optimizer = optim(
            resnet.parameters(),
            **params,
        )

        input.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            input.optimizer, [200, 350], 0.1
        )

        output = train_network(
            resnet,
            input,
        )

        timestamp = get_unique_id()

        save_network_output(
            path="./tests/optims/outputs",
            input=output,
            custom_name=f"resnet-{name}-{timestamp}",
        )

        save_network_loss_graph(
            path="./tests/optims/graphs",
            input=output,
            custom_name=f"resnet-{name}-{timestamp}",
        )

        save_network_test_results(
            path="./tests/optims/results",
            input=test_network(
                network=resnet,
                input=Network.TestingInput(
                    testloader=testloader,
                    device=device,
                ),
            ),
            custom_name=f"resnet-{name}-{timestamp}",
        )
