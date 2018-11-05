from parse_config import get_configuration
from models.image_features_dataset import ImageFeaturesDataset
from models.train_predict import predict as predict_all
from debug_helpers import format_duration

import torch
from torch.utils import data
from datetime import datetime


def generate_predictions(filepaths, config):
    model = torch.load(filepaths['model'])
    text_mapper = torch.load(filepaths['vocab'])
    
    # generate predictions for: test, val and train data
    for split in ['test', 'val', 'train']:
        predicted_sentences = predict(
            filepaths[f'image_features_{split}'], model, text_mapper,
            config['max_length'], config['dl_params']
        )
        write_lines(predicted_sentences, filepaths[f'predictions_{split}'])
    

def predict(fpath_imfeats, model, text_mapper,
            max_length, dl_params):

    start = datetime.now()

    # create models and data
    global_feats = type(model).__name__ == 'ShowTell'
    data_set = ImageFeaturesDataset(fpath_imfeats, global_feats)
    data_loader = data.DataLoader(data_set, **dl_params)

    # predict captions
    SOS_index = text_mapper.SOS_index()
    EOS_index = text_mapper.EOS_index()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_indices = predict_all(model, data_loader, SOS_index, max_length, device)
    
    # convert indices to words
    predicted_sentences = [text_mapper.indices2sentence(s) for s in predicted_indices]
        
    end = datetime.now()
    print(f'{len(predicted_sentences)} sentences predicted in {format_duration(start, end)}.')
    
    return predicted_sentences

def write_lines(lines, fpath):
    with open(fpath_save_predictions, "w") as f:
        f.write('\n'.join(predicted_sentences))
            
def main():
    filepaths, config = get_configuration(
        'predict', 
        description = 'Generate image captions.')
    generate_predictions(filepaths, config)


if __name__ == "__main__":
    main()
     
