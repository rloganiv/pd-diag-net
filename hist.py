import argparse
import matplotlib.pyplot as plt
import pdb
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='hist filepath', type=str)
args = parser.parse_args()


def plot_loss(hist_dict):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    losses = [hist_dict[i]['loss'] for i in xrange(10)]
    val_losses = [hist_dict[i]['val_loss'] for i in xrange(10)]
    for loss in losses:
        ax.plot(loss, color='blue', alpha=0.75)
    ax.plot(loss, color='blue', alpha=0.75, label='Train Loss')
    for loss in val_losses:
        ax.plot(loss, color='orange', alpha=0.75)
    ax.plot(loss, color='orange', alpha=0.75, label='Validation Loss')
    ax.legend()
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

