from evaluation.plots import plot_epoch_losses, plot_epoch_blue_scores
import evaluation.blue_calculator as blue
from parse_config import get_configuration

import torch

def evaluate(filepaths):
    calculate_blue(filepaths)
    plot_epoch_metrics(filepaths)

def plot_epoch_metrics(filepaths):
    epoch_metrics = torch.load(filepaths['epoch_metrics'])
    plot_epoch_losses(
        epoch_metrics['train_losses'], 
        epoch_metrics['val_losses'], 
        filepaths['plot_epoch_loss']
    )

    plot_epoch_blue_scores(
        epoch_metrics['val_blue_scores'], 
        filepaths['plot_epoch_bleu']
    )
    
def calculate_blue(filepaths):
    for split in ['val', 'train', 'test']:
        blue.run_multi_bleu(
            filepaths[f'captions_{split}'], 
            filepaths[f'predictions_{split}'], 
            filepaths[f'bleu_{split}']
        )
        blue.run_multi_blue_compare_human_performance(
            filepaths[f'captions_{split}'], 
            filepaths[f'predictions_{split}'], 
            filepaths[f'bleu_{split}']
        )

def main():
    filepaths, _ = get_configuration(
        'evaluate', 
        description = 'Generates captions for images.')
    evaluate(filepaths)

if __name__ == "__main__":
    main()
     
