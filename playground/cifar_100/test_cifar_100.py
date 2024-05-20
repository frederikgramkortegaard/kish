import os
import time
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Models
from modules.Optimizer import AdamP, AdamW, OrthAdam
from torchvision.models.densenet import densenet121
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnext50_32x4d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    argparse = argparse.ArgumentParser()
    argparse.add_argument("-debug", action="store_true")
    argparse.add_argument(
        "-datapath",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/data",
    )
    argparse.add_argument(
        "-checkpointpath",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/checkpoints",
    )
    argparse.add_argument(
        "-resultspath",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/results",
    )
    argparse.add_argument("-device", type=str, default=device, choices=["cpu", "cuda"])
    argparse.add_argument("-epochs", type=int, default=186)
    argparse.add_argument("-disable_transforms", action="store_true", default=False)
    args = argparse.parse_args()

    if not os.path.isdir(args.resultspath):
        os.mkdir(args.resultspath)

    if not os.path.isdir(args.checkpointpath):
        os.mkdir(args.checkpointpath)

    if args.disable_transforms:
        if args.debug:
            print("Disabling transforms")

        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        if args.debug:
            print("Using transforms")
        # Define the transformation for the dataset
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    # Load the CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(
        root=args.datapath,
        train=True,
        download=True,
        transform=train_transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR100(
        root=args.datapath,
        train=False,
        download=True,
        transform=test_transform,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )

    # Setup Models
    for optim, name in zip([AdamW, AdamP, OrthAdam], ["AdamW", "AdamP", "OrthAdam"]):

        if args.debug:
            print(f"Training with optimizer: {name}")

        # Setup DenseNet
        densenet = densenet121(num_classes=10)
        densenet_criterion = nn.CrossEntropyLoss()
        densenet_optimizer = optim(densenet.parameters(), lr=0.003, weight_decay=3e-4)
        densenet_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            densenet_optimizer, [81, 122], 0.1
        )

        # Setup ResNet
        resnet = resnet18(num_classes=10)
        resnet_criterion = nn.CrossEntropyLoss()
        resnet_optimizer = optim(resnet.parameters(), lr=0.003, weight_decay=3e-4)
        resnet_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            resnet_optimizer, [81, 122], 0.1
        )

        # Setup ResNext
        resnext = resnext50_32x4d(num_classes=10)
        resnext_criterion = nn.CrossEntropyLoss()
        resnext_optimizer = optim(resnext.parameters(), lr=0.003, weight_decay=3e-4)
        resnext_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            resnext_optimizer, [81, 122], 0.1
        )

        # Train the models. The following train/test code could be made more generic instead of repeat code, but for ease-of-verification we will keep it as is

        ## train densenet121
        if args.debug:
            print("\tTraining DenseNet121")
        densenet.to(device)

        densenet_losses = []
        densenet_accuracies = []
        for epoch in range(args.epochs):
            if args.debug:
                print(f"\t\tEpoch: {epoch}")
            densenet.train()
            for i, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                densenet_optimizer.zero_grad()
                outputs = densenet(inputs)
                loss = densenet_criterion(outputs, targets)
                densenet_losses.append(loss.item())

                loss.backward()
                densenet_optimizer.step()
            densenet_scheduler.step()

            # test densenet121
            if args.debug:
                print("\tTesting DenseNet121")
            densenet.to(device)
            densenet.eval()
            densenet_correct = 0
            densenet_total = 0
            for i, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = densenet(inputs)
                _, predicted = torch.max(outputs, 1)
                densenet_total += targets.size(0)
                densenet_correct += (predicted == targets).sum().item()

            densenet_accuracy = densenet_correct / densenet_total
            densenet_accuracies.append(densenet_accuracy)
            if args.debug:
                print(f"\t\tDenseNet121 Accuracy: {densenet_accuracy}")

        ## train resnet18
        if args.debug:
            print("\tTraining ResNet18")
        resnet.to(device)

        resnet_losses = []
        resnet_accuracies = []
        for epoch in range(args.epochs):
            if args.debug:
                print(f"\t\tEpoch: {epoch}")
            resnet.train()
            for i, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                resnet_optimizer.zero_grad()
                outputs = resnet(inputs)
                loss = resnet_criterion(outputs, targets)
                resnet_losses.append(loss.item())

                loss.backward()
                resnet_optimizer.step()
            resnet_scheduler.step()

            # test resnet18
            if args.debug:
                print("\tTesting ResNet18")
            resnet.to(device)
            resnet.eval()
            resnet_correct = 0
            resnet_total = 0
            for i, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = resnet(inputs)
                _, predicted = torch.max(outputs, 1)
                resnet_total += targets.size(0)
                resnet_correct += (predicted == targets).sum().item()

            resnet_accuracy = resnet_correct / resnet_total
            resnet_accuracies.append(resnet_accuracy)
            if args.debug:
                print(f"\t\tResNet18 Accuracy: {resnet_accuracy}")

        ## train resnext50_32x4d
        if args.debug:
            print("\tTraining ResNext50")
        resnext.to(device)

        resnext_losses = []
        resnext_accuracies = []
        for epoch in range(args.epochs):
            if args.debug:
                print(f"\t\tEpoch: {epoch}")
            resnext.train()
            for i, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                resnext_optimizer.zero_grad()
                outputs = resnext(inputs)
                loss = resnext_criterion(outputs, targets)
                resnext_losses.append(loss.item())

                loss.backward()
                resnext_optimizer.step()
            resnext_scheduler.step()

            # test resnext50_32x4d
            if args.debug:
                print("\tTesting ResNext50")
            resnext.to(device)
            resnext.eval()
            resnext_correct = 0
            resnext_total = 0
            for i, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = resnext(inputs)
                _, predicted = torch.max(outputs, 1)
                resnext_total += targets.size(0)
                resnext_correct += (predicted == targets).sum().item()

            resnext_accuracy = resnext_correct / resnext_total
            resnext_accuracies.append(resnext_accuracy)
            if args.debug:
                print(f"\t\tResNext50 Accuracy: {resnext_accuracy}")

        # Save the results
        end_time = time.time()

        torch.save(
            {
                "densenet_losses": densenet_losses,
                "densenet_accuracies": densenet_accuracies,
                "resnet_losses": resnet_losses,
                "resnet_accuracies": resnet_accuracies,
                "resnext_losses": resnext_losses,
                "resnext_accuracies": resnext_accuracies,
            },
            f"{args.resultspath}/{name}_{end_time}{'_no_augments' if args.disable_transforms else '_with_augments'}.pt",
        )   

        if args.epochs != 0:
            print(
                name,
                max(densenet_accuracies),
                max(resnet_accuracies),
                max(resnext_accuracies),
            )
    if args.debug:
        print("Program finished...")
