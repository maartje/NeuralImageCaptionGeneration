from nltk.translate.bleu_score import corpus_bleu
from models.train_predict import predict

import torch

class BlueCalculator:

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
        
        

