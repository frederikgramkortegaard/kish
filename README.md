# Kish: A Comprehensive Deep- & Reinforcement Learning Testing & Evaluation Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/frederikgramkortegaard/kish/blob/master/LICENSE)
[![CodeQL](https://github.com/frederikgramkortegaard/kish/workflows/CodeQL/badge.svg)](https://github.com/frederikgramkortegaard/kish/actions?query=workflow%3ADependency+Review)
[![Dependency Review](https://github.com/frederikgramkortegaard/kish/workflows/Dependency%20Review/badge.svg)]()

## About
Comprehensive Deep- & Reinforcement Learning Testing & Evaluation Framework. Designed with a Pythonic and object-oriented approach, Kish offers an efficient platform for testing and evaluating diverse neural network architectures. Through interfaces and adapters, integrating pre-made models becomes effortlessly plug-and-play. Embracing both iterative live updates and static analysis, Kish allows researchers and practitioners in the field to seamlessly fine-tune models.

## Installation
## Command Line Interface
```bash
usage: main.py [-h] [-c CONFIG_FILE] [-r] [-sr]

Kish: A PyTorch-based neural network testing & Evaluation framework

options:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config CONFIG_FILE
                        Configuration file to use (default: config.ini)
  -r, --render          Render the environment for Reinforcement Learning Training (default: False)
```

## Training
### Model Requirements
#### Reinforcement Agents


## Graphing
### Rendering & Live Updates
Kish offers a live rendering of the environment during training. This allows for a quick overview of the training process and the ability to spot potential issues early on. The ,method `live_report_reinforcement_agent` takes in a `generator` returned from `iterative_train_reinforcement_agent` and creates a graph showcasing the reward pr. episode with a moving average. Further, the `--render` flag can be used to render the environment during training using the `gym` package. It should be clearly noted that using either (or both) of these flags significantly slow down the training process. If you're looking to _emulate_ a live_report using a pre-made `Reinforcement.TrainingOutput` see `emulate_iterative_renforcement_agent_training`. 


## Troubleshooting
If you have any issues with the application, please open an issue to discuss the problem you are facing.

## License
This project uses the [MIT License](https://choosealicense.com/licenses/mit/).
