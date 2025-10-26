# AttentionSplit: A Hybrid Neural Network Architecture for Long- and Short-Term Retention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

:pushpin: [Thesis](https://github.com/frederikgramkortegaard/kish/blob/master/Thesis.pdf)
:pushpin: [Code Examples](https://github.com/frederikgramkortegaard/kish/tree/master/playground)

**AttentionSplit** is a hybrid neural network architecture designed to enhance long- and short-term retention in temporal data tasks. 

This project also includes **OrthAdam**, a novel Adam optimizer variant inspired by AdamP, aimed at improving convergence and maximum achievable accuracy. The framework has been tested across reinforcement learning, image classification, and sequence prediction tasks.

For full details, refer to our Thesis which describes the theory, experiments, and evaluation in depth.

---

## Features

- **AttentionSplit Module**: Combines recurrence (like LSTMs) with attention mechanisms to improve temporal representation.
- **OrthAdam Optimizer**: A modified Adam variant with projection criteria changes to improve optimization performance.
- **Flexible Testing**: Supports OpenAI Gymnasium classic control, Mujoco continuous control, and image classification datasets.
- **Reproducibility**: Includes example scripts and logging to allow replication of experiments.

---

## Installation

1. Install Python dependencies:

    pip install -r requirements.txt

2. Install the Mujoco physics engine (required for certain RL environments):

    https://github.com/openai/mujoco-py?tab=readme-ov-file

---

## Usage

All experiments can be found in the `playground` folder. You can run individual tests or use the main runner:

    # Example: Run a Mujoco HalfCheetah test
    python playground/run.py --env HalfCheetah-v4 --model AttentionSplit

Example using OrthAdam in a training loop:

    from modules.Optimizer import OrthAdam
    from modules.Attentionsplit import AttentionSplitModule
    import torch

    model = AttentionSplitModule(input_dim=32, hidden_dim=64, output_dim=10)
    optimizer = OrthAdam(model.parameters(), lr=0.001)

    # Forward pass and optimization
    loss = model(torch.randn(16, 32))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

---

## Datasets and Environments

- OpenAI Gymnasium – Classic Control: CartPole, MountainCar  
- OpenAI Mujoco – Continuous Control: HalfCheetah, Walker2D  
- Image Classification: CIFAR-10, CIFAR-100, FashionMNIST  
- Sequence Prediction: Custom temporal datasets  

---

## Testing and Benchmarking

You can explore tests and benchmarking using the provided scripts in the `playground` folder. For options:

    python playground/run.py --help

---

## Results Overview

Experiments indicate:

- **OrthAdam** achieves higher maximal accuracy than standard Adam and AdamP across multiple image classification tasks.  
- **AttentionSplit** shows improved performance for temporal data analysis compared to standard LSTM or pure attention models in some RL environments.  

> Refer to [Section 9 of the Thesis](Thesis.pdf) for detailed results, graphs, and evaluation metrics.

---

## Future Work

- Scale AttentionSplit to NLP and video sequence tasks.  
- Optimize OrthAdam further for large-scale RL environments.  
- Explore multi-modal datasets and hybrid training regimes.  

---

## License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
