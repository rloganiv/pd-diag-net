import argparse
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='hist filepath', type=str)
parser.add_argument('-t', '--title', help='plot title', type=str)
parser.add_argument('-y', '--ymax', help='max y-value', type=float)
args = parser.parse_args()


def plot_loss(hist_dict):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    losses = [hist_dict[i]['loss'] for i in xrange(10)]
    mean_loss = np.mean(np.array(losses), axis=0)
    val_losses = [hist_dict[i]['val_loss'] for i in xrange(10)]
    mean_val_loss = np.mean(np.array(val_losses), axis=0)
    for loss in losses:
        ax.plot(loss, color='blue', alpha=0.25)
    ax.plot(mean_loss, color='blue', label='Train Loss')
    for loss in val_losses:
        ax.plot(loss, color='orange', alpha=0.25)
    ax.plot(mean_val_loss, color='orange', label='Validation Loss')
    ax.set_ylim(ymin=0)
    if args.ymax:
        ax.set_ylim(ymax=args.ymax)
    ax.legend()
    if args.title:
        ax.set_title(args.title)
        plt.savefig(args.title + '.png')
    else:
        plt.savefig(args.path + '.png')


def accuracy(hist_dict):
    acc = [hist_dict[i]['val_acc'][-1] for i in xrange(10)]
    print 'Validation accuracy: %f' % (sum(acc)/10.)


if __name__ == '__main__':
    print args.path
    with open(args.path, 'rb') as pkl_file:
        hist_dict = pickle.load(pkl_file)
    plot_loss(hist_dict)
    accuracy(hist_dict)

