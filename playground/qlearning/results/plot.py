import matplotlib.pyplot as plt
import torch
import argparse
import os
import time


def generate_graph(dataset, name, show, save, graphpath, output_name=None, title=None):


    print(list(dataset.keys()))

    fig, axs = plt.subplots(3)
    print(len(dataset['attention']))
    print(len(dataset['lstm']))
    print(len(dataset['transformer']))
    # set figure title
    fig.suptitle("MountainCar-v0 Accuracy Evaluation", fontsize=16)
    # sub subtitle
    fig.text(0.5, 0.94, "(128 Batch Size - 328 Batches - Averaged over 5 Runs)", ha='center')    

    # print the min and max values
    print(f"Min value: {min(dataset['attention'])}")
    print(f"Max value: {max(dataset['attention'])}")

    print(f"Min value: {min(dataset['lstm'])}")
    print(f"Max value: {max(dataset['lstm'])}")

    print(f"Min value: {min(dataset['transformer'])}")
    print(f"Max value: {max(dataset['transformer'])}")


# 
    axs[0].plot([x*100 for x in dataset['attention']], label="AttentionSplit")
    axs[1].plot([x*100 for x in dataset['lstm']], label="LSTM")
    axs[2].plot([x*100 for x in dataset['transformer']], label="Transformer")

    attentionsplit_total_mean = sum(dataset['attention']) / len(dataset['attention'])
    lstm_total_mean = sum(dataset['lstm']) / len(dataset['lstm'])
    transformer_total_mean = sum(dataset['transformer']) / len(dataset['transformer'])

    print(f"AttentionSplit Total Mean: {attentionsplit_total_mean*100:.2f}")
    print(f"LSTM Total Mean: {lstm_total_mean*100:.2f}")
    print(f"Transformer Total Mean: {transformer_total_mean*100:.2f}")

    # set x and y label
    for ax in axs:
        ax.set_xlabel("Test Batch #")
        ax.set_ylabel("Accuracy (%)")


    ## Add name above plot
    for ax, name in zip(axs, ['AttentionSplit', 'LSTM', 'Transformer']):
        ax.set_title(name)

    # Set y ticks force each tick to be in every plot
    for ax in axs:
        ax.set_ylim(95, 100)  # Set y-axis limits
        ax.set_yticks(range(95, 101, 1), minor=False)

    # Increase vertical space between subplots
    plt.subplots_adjust(hspace=0.5)

    # Calculate mean over time (rolling window 10)
    attention_mean = [sum(dataset['attention'][i:i+10])/10 for i in range(len(dataset['attention'])-10)]
    lstm_mean = [sum(dataset['lstm'][i:i+10])/10 for i in range(len(dataset['lstm'])-10)]
    transformer_mean = [sum(dataset['transformer'][i:i+10])/10 for i in range(len(dataset['transformer'])-10)]


    # Add mean line
    #axs[0].plot(range(10, len(attention_mean)+10), [x*100 for x in attention_mean], label=f'Mean ({attentionsplit_total_mean*100:.2f}%)', linestyle='--')
    #axs[1].plot(range(10, len(lstm_mean)+10), [x*100 for x in lstm_mean], label=f'Mean ({lstm_total_mean*100:.2f}%)', linestyle='--')
    #axs[2].plot(range(10, len(transformer_mean)+10), [x*100 for x in transformer_mean], label=f'Mean ({transformer_total_mean*100:.2f}%)', linestyle='--')

    # add a line at the mean
    for ax, mean in zip(axs, [attentionsplit_total_mean, lstm_total_mean, transformer_total_mean]):
        ax.axhline(y=mean*100, color='r', linestyle='--', label=f'Mean ({mean*100:.2f}%)')
    

    # Change legend name
    axs[0].legend(loc="upper left")
    axs[1].legend(loc="upper left")
    axs[2].legend(loc="upper left")


    if save:
        print(f"Saving graph to {graphpath}/{output_name}_accuracy.png")
        fig.savefig(f'{graphpath}/{output_name}_accuracy.png')

    if show:
        plt.show()

    fig.clf()

if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--graph_path', type=str, default='graphs')
    argparser.add_argument('--show', action='store_true')
    argparser.add_argument('--save', action='store_true')
    argparser.add_argument('--results', type=str, default='.', help='Path to .pt files, default is .')
    argparser.add_argument('--title', type=str, default=None)
    args = argparser.parse_args()
    
    if not os.path.exists(args.graph_path):
        os.makedirs(args.graph_path)

    if not os.path.exists(args.results):
        print(f"Path to results: {args.results} does not exist")
        exit()

    results = {}
    for file in os.listdir(args.results):
        if file.endswith('.pt'):
            results[file] = torch.load(os.path.join(args.results, file))



    ##### TRAINED ON 5 EPOCHS OF ALL THE TEST DATA (OUTPUT FROM DEEP QLEARNING)

    ## Get the mean of all Acrobot-v1_ files
    acrobot = {
        'attention': [],
        'lstm': [],
        'transformer': []
    }

    cartpole = {
        'attention': [],
        'lstm': [],
        'transformer': []
    }

    mountaincar = {
        'attention': [],
        'lstm': [],
        'transformer': []
    }

    for key, value in results.items():
        if 'Acrobot-v1' in key:
            acrobot['attention'].append(value['attention'])
            acrobot['lstm'].append(value['lstm'])
            acrobot['transformer'].append(value['transformer'])

        if 'CartPole-v1' in key:
            cartpole['attention'].append(value['attention'])
            cartpole['lstm'].append(value['lstm'])
            cartpole['transformer'].append(value['transformer'])

        if 'MountainCar-v0' in key:
            mountaincar['attention'].append(value['attention'])
            mountaincar['lstm'].append(value['lstm'])
            mountaincar['transformer'].append(value['transformer'])

    ## Get the mean of all accuracies across all the files
    acrobot_means = {
        'attention': [sum(x) / len(x) for x in zip(*acrobot['attention'])],
        'lstm': [sum(x) / len(x) for x in zip(*acrobot['lstm'])],
        'transformer': [sum(x) / len(x) for x in zip(*acrobot['transformer'])]
    }

    cartpole_means = {
        'attention': [sum(x) / len(x) for x in zip(*cartpole['attention'])],
        'lstm': [sum(x) / len(x) for x in zip(*cartpole['lstm'])],
        'transformer': [sum(x) / len(x) for x in zip(*cartpole['transformer'])]
    }

    mountaincar_means = {
        'attention': [sum(x) / len(x) for x in zip(*mountaincar['attention'])],
        'lstm': [sum(x) / len(x) for x in zip(*mountaincar['lstm'])],
        'transformer': [sum(x) / len(x) for x in zip(*mountaincar['transformer'])]
    }


    generate_graph(mountaincar_means, 'MountainCar-v0', args.show, args.save, args.graph_path, 'MountainCar-v0', args.title)
