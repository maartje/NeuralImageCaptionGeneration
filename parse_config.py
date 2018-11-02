import json
import argparse
import os
import glob

def parse_args(section, description):
    parser = argparse.ArgumentParser(
        description = description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    options(parser, section)
    opt = parser.parse_args()
    return opt

def options(parser, section):
    parser.add_argument(
        '--config', 
        help = "Path to config file in JSON format",
        default = 'config.json')
    if section == 'preprocess':
        parser.add_argument(
            '--min_occurences', 
            type = float,
            help = "Overwrites the config setting for the minimum occurrence of a word to be included in the vocabulary")
    if section == 'train':
        parser.add_argument(
            '--learning_rate', 
            type = float,
            help = "Overwrites the config setting for the learning rate")
        parser.add_argument(
            '--optimizer', 
            help = "Overwrites the config setting for the optimizer")
        parser.add_argument(
            '--model', 
            help = "Overwrites the config setting for the model type, i.e. 'show_tell', 'show_attend_tell'.")
        parser.add_argument(
            '--alpha_c', 
            type = float,
            help = "Overwrites the config setting for the alpha regulizer in the 'show_attend_tell' model")
        
def load_config(fpath_config):
    with open(fpath_config) as f:
        config = json.load(f)
    return config

def get_configuration(section, description = ''):
    opt = parse_args(section, description)
    config = load_config(opt.config)

    # overwrite config settings with commandline arguments    
    opt_dict = vars(opt)
    opt_dict = {k : v for k, v in opt_dict.items() if not (v is None)}
    config[section].update({ k:v for k, v in opt_dict.items() if k in config[section]})
    
    filepaths = get_file_paths(config['input'])
    return config[section], filepaths

def get_file_paths(config):
    main_dir = "data"
    input_dir = os.path.join(main_dir, 'input')
    preprocess_dir = os.path.join(main_dir, 'preprocess')
    train_dir = os.path.join(main_dir, 'train')
    predict_dir = os.path.join(main_dir, 'predict')
    evaluate_dir = os.path.join(main_dir, 'evaluate')

    # input
    fpattern_captions_train = os.path.join(input_dir, config['fpattern_captions_train'])
    fpattern_captions_val = os.path.join(input_dir, config['fpattern_captions_val'])
    fpattern_captions_test = os.path.join(input_dir, config['fpattern_captions_test'])
    fpaths_captions_train = glob.glob(fpattern_captions_train)
    fpaths_captions_val = glob.glob(fpattern_captions_val)
    fpaths_captions_test = glob.glob(fpattern_captions_test)

    # preprocess
    to_vector_path = lambda fp: f"{fp.replace('input', 'preprocess')}.pt"
    fpaths_caption_vectors_train = [to_vector_path(fp) for fp in fpaths_captions_train]
    fpaths_caption_vectors_val = [to_vector_path(fp) for fp in fpaths_captions_val]
    fpaths_caption_vectors_test = [to_vector_path(fp) for fp in fpaths_captions_test]
    fpath_vocab = os.path.join(preprocess_dir, 'vocab.pt')
        
    # train
    fpath_im_features_train = os.path.join(
        preprocess_dir, config['fname_image_features_train'])
    fpath_im_features_val = os.path.join(
        preprocess_dir, config['fname_image_features_val'])
    fpath_im_features_test = os.path.join(
        preprocess_dir, config['fname_image_features_test'])
    fpath_losses = os.path.join(train_dir, 'losses.pt')
    fpath_bleu_scores = os.path.join(train_dir, 'bleu_scores.pt')
    fpath_model = os.path.join(train_dir, f'model.%d.pt')
    fpath_best_model = os.path.join(train_dir, f'model_best.pt')

    # predict
    fpath_predictions_val = os.path.join(predict_dir, 'predictions_val.txt')
    fpath_predictions_test = os.path.join(predict_dir, 'predictions_test.txt')
    fpath_predictions_train = os.path.join(predict_dir, 'predictions_train.txt')
    
    # evaluate
    fpath_plot_epoch_loss = os.path.join(evaluate_dir, 'epoch_losses.png')
    fpath_plot_batch_loss = os.path.join(evaluate_dir, 'batch_losses.png')
    fpath_plot_bleu = os.path.join(evaluate_dir, 'epoch_BLEU.png')
    fpath_bleu_val = os.path.join(evaluate_dir, 'BLEU_val.txt')
    fpath_bleu_test = os.path.join(evaluate_dir, 'BLEU_test.txt')
    fpath_bleu_train = os.path.join(evaluate_dir, 'BLEU_train.txt')
    fpath_bleu_human_test = os.path.join(evaluate_dir, 'BLEU_test_human_comparison.txt')
        
    return {
        # input
        'captions_train' : fpaths_captions_train,
        'captions_val' : fpaths_captions_val,
        'captions_test' : fpaths_captions_test,
        'image_features_train' : fpath_im_features_train,
        'image_features_val': fpath_im_features_val,
        'image_features_test': fpath_im_features_test,
                
        # preprocess
        'vocab' : fpath_vocab,
        'caption_vectors_train' : fpaths_caption_vectors_train,
        'caption_vectors_val' : fpaths_caption_vectors_val,
        'caption_vectors_test' : fpaths_caption_vectors_test,
        
        # train
        'losses' : fpath_losses,
        'bleu_scores' : fpath_bleu_scores,
        'model' : fpath_model,
        'best_model' : fpath_best_model,
        
        # predict
        'predictions_test' : fpath_predictions_test,
        'predictions_val' : fpath_predictions_val,
        'predictions_train' : fpath_predictions_train,
        
        # evaluate
        'plot_epoch_loss' : fpath_plot_epoch_loss,
        'plot_batch_loss' : fpath_plot_batch_loss,
        'plot_bleu' : fpath_plot_bleu,
        'bleu_val' : fpath_bleu_val,
        'bleu_test' : fpath_bleu_test,
        'bleu_train' : fpath_bleu_train,
        'bleu_human_test' : fpath_bleu_human_test
    }

