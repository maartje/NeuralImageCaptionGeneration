from parse_config import get_configuration
from models.show_tell import ShowTell
from models.image_captions_dataset import ImageCaptionsDataset, collate_image_captions
from models.train_predict import fit
from evaluation.metrics_collector import MetricsCollector
from evaluation.train_output_writer import TrainOutputWriter
from evaluation.model_saver import ModelSaver

from torch.utils import data
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch import optim

def train(filepaths, config):
    update_config(filepaths, config)
    model = configure_model(filepaths, config)
    dl_train = configure_data_loader(filepaths, config, split='train')
    dl_val = configure_data_loader(filepaths, config, split='val')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.NLLLoss(ignore_index = config['PAD_index'])
    model.to(device)
    loss_criterion.to(device)
    optimizer = configure_optimizer(config, model)
    if config.get('clip'):
        clip_grad_norm_(model.parameters(), config['clip'])
    
    metrics_collector = MetricsCollector(
        model, dl_val, config['max_length'], loss_criterion, device
    )
    metrics_collector.store_val_metrics() # store initial validation loss and BLUE
    trainOutputWriter = TrainOutputWriter(metrics_collector)
    modelSaver = ModelSaver(model, metrics_collector, filepaths['model'])
    fn_epoch_listeners = [
        metrics_collector.store_train_loss,
        metrics_collector.store_val_metrics,
        trainOutputWriter.print_epoch_info,
        modelSaver.save_best_model
    ]
    
    fit(model, dl_train, loss_criterion, optimizer, 
        config['epochs'], device,
        fn_epoch_listeners, config.get('alpha_c'))
            
    # save metrics collected during training
    torch.save({
        'train_losses': metrics_collector.train_losses,
        'val_losses': metrics_collector.val_losses,
        'val_blue_scores': metrics_collector.val_blue_scores
    }, filepaths['epoch_metrics'])



def configure_optimizer(config, model):
    if config['optimizer'] == 'ADAM':
        return optim.Adam(model.parameters(), lr = config['learning_rate'])
    if config['optimizer'] == 'SGD':
        return optim.SGD(model.parameters(), lr = config['learning_rate'])
    raise ValueError(f"optimizer type {config['optimizer']} is not supported.")

def configure_data_loader(filepaths, config, split='train'):
    global_feats = config['model'] == "show_tell"
    dataset = ImageCaptionsDataset(
        filepaths[f'image_features_{split}'], 
        filepaths[f'caption_vectors_{split}'], 
        global_feats
    )
    collate_fn = lambda b: collate_image_captions(b, config['PAD_index'])
    dl_params = config[f"dl_params_{split}"]
    dl_params['collate_fn'] = collate_fn
    dataloader = data.DataLoader(dataset, **dl_params)
    return dataloader

def configure_model(filepaths, config):
    if config['model'] == "show_tell":
        return ShowTell(
            config['encoding_size'], 
            config['hidden_size'], 
            config['vocab_size'], 
            config['PAD_index'],
            config['dropout']
        )
    if config['model'] == "show_attend_tell":
        return DecoderWithAttention(
            config['hidden_size'], 
            config['hidden_size'], 
            config['hidden_size'], 
            config['vocab_size'], 
            config['encoding_size'], 
            config['dropout']
        )
    raise ValueError(f'model type {model} is unknown.')

def update_config(filepaths, config):
    text_mapper = torch.load(filepaths['vocab'])
    config['vocab_size'] = text_mapper.vocab.n_words
    config['PAD_index'] = text_mapper.PAD_index()
             
def main():
    filepaths, config = get_configuration(
        'train', 
        description = 'Train model for generating image descriptions.')
    train(filepaths, config)


if __name__ == "__main__":
    main()
     
