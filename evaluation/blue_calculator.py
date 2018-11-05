from nltk.translate.bleu_score import corpus_bleu
from models.train_predict import predict

import torch
from os import system

class BlueCalculator:
    """ Calculates BLUE scores using the nltk implementation of corpus BLUE.
        Use this class to calculate BLUE scores during training
    """

    def __init__(self, model, text_mapper, val_data, references, max_length, device):
        self.model = model
        self.text_mapper = text_mapper
        self.val_data = val_data
        self.references = self.reorder_reference_sentences(references)
        self.max_length = max_length
        self.device = device
        
    def calculate_blue(self):
        predicted_indices = predict(
            self.model, self.val_data, self.text_mapper.SOS_index(), 
            self.max_length, self.device)
        cleaned_predicted_indices = [
            self.text_mapper.remove_predefined_indices(s) for s in predicted_indices
        ]
        return corpus_bleu(self.references, cleaned_predicted_indices)
        
    def reorder_reference_sentences(self, references_by_file):
        references_by_sentence = [
            [ s[1:-1] for s in tup] for tup in zip(*references_by_file)
        ]
        return references_by_sentence


        
def run_multi_bleu(fpaths_references, fpath_predicted, fpath_out):
    """ Calculates BLUE scores using the official 'muli-blue.perl script'.
        Use this function to calculate the final BLUE scores on test data
    """

    fpaths_references_str = ' '.join(fpaths_references)
    system(
        f'./scripts/multi-bleu.perl -lc {fpaths_references_str} < {fpath_predicted} > {fpath_out}'
    )
    
def run_multi_blue_compare_human_performance(fpaths_references, fpath_predicted, fpath_out):
    """ Compares BLUE scores for model predictions and for human captions'.
        The blue scores are calculated by comparing one reference file with the other
        reference files (human), and by comparing the model predictions with all exept one
        reference files (model).
    """

    # human performance
    fpaths_references_str = ' '.join(fpaths_references[:-1])
    fpath_human = fpaths_references[-1]
    system(f'echo BLEU score human annotator based on {len(fpaths_references[:-1])} references >> {fpath_out}')
    system(
        f'./scripts/multi-bleu.perl -lc {fpaths_references_str} < {fpath_human} >> {fpath_out}'
    )
    
    system(f'echo >> {fpath_out}')
    
    # model performance
    system(f'echo BLEU score model based on {len(fpaths_references[:-1])} references >> {fpath_out}')
    system(
        f'./scripts/multi-bleu.perl -lc {fpaths_references_str} < {fpath_predicted} >> {fpath_out}'
    )

