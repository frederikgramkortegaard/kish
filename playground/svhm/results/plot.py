import matplotlib.pyplot as plt
import torch
import argparse
import os
import time



def generate_graph(dataset, name, show, save, graphpath, output_name=None, title=None):

    if output_name == None:
        output_name = name


    # plot on the first subplot
    plt.suptitle(f'Accuracy per Epoch ({name})')
    if title is not None:
        plt.title(title)

    resnet_label = "Resnet18              (Max: {:.2f}%)".format(max(dataset['resnet_accuracies'])*100)
    densenet_label = "Densenet121        (Max: {:.2f}%)".format(max(dataset['densenet_accuracies'])*100)
    resnext_label = "ResNext50_32x4d (Max: {:.2f}%)".format(max(dataset['resnext_accuracies'])*100)
    
    plt.plot([x*100 for x in dataset['resnet_accuracies']], label=resnet_label)
    plt.plot([x*100 for x in dataset['densenet_accuracies']], label=densenet_label)
    plt.plot([x*100 for x in dataset['resnext_accuracies']], label=resnext_label)

    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')

    plt.minorticks_on()
    plt.legend()

    if save:
        print(f"Saving graph to {graphpath}/{output_name}_accuracy.png")
        plt.savefig(f'{graphpath}/{output_name}_accuracy.png')

    if show:
        plt.show()

    plt.clf()


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--graph_path', type=str, default='graphs')
    argparser.add_argument('--show', action='store_true')
    argparser.add_argument('--save', action='store_true')
    argparser.add_argument('--results', type=str, default='.', help='Path to .pt files, default is ./results')
    argparser.add_argument('--title', type=str, default=None)
    args = argparser.parse_args()

    if not os.path.exists(args.graph_path):
        os.makedirs(args.graph_path)

    if not os.path.exists(args.results):
        print(f"Path to results: {args.results} does not exist")
        exit()

    for file in os.listdir(args.results):
        if file.endswith('.pt'):
            results = torch.load(os.path.join(args.results, file))
            generate_graph(results, file.split('_')[0], args.show, args.save, args.graph_path, output_name=file+'_'+str(time.time()),title="")

