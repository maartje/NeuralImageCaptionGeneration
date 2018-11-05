import matplotlib
import itertools
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_epoch_losses(train_losses, val_losses, fname, title = None):
    """Plot training and validation loss for each epoch.""" 
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'ro-', color='blue', label='train loss')
    plt.plot(val_losses, 'ro-', color='red', label='validation loss')
    plt.xlabel('#epochs')
    plt.ylabel('avg. loss')
    plt.legend()
    if title:
        plt.title(title)
    plt.savefig(fname)
    plt.close()

def plot_epoch_blue_scores(val_blue_scores, fname, title = None):
    """Plot blue scores for each epoch.""" 
    plt.plot(val_blue_scores, 'ro-', color='red', label='BLEU on validation')
    plt.xlabel('#epochs')
    plt.ylabel('BLUE')
    plt.legend()
    if title:
        plt.title(title)
    plt.savefig(fname)
    plt.close()


