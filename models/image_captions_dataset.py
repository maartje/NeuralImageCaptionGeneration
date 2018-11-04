import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import tables
from torch.utils import data
import math

class ImageCaptionsDataset(data.Dataset):

    def __init__(self, fpath_image_features, fpaths_description_vectors, global_feats = True):
        super(ImageCaptionsDataset, self).__init__()
        
        # image features
        self.fpath_image_features = fpath_image_features
        self.h5file = tables.open_file(fpath_image_features, mode="r")
        if global_feats :
            self.im_feats = self.h5file.root.global_feats
        else :
            self.im_feats = self.h5file.root.local_feats
        # descriptions
        self.description_lists = [torch.load(fpath) for fpath in fpaths_description_vectors]
        

    def __len__(self):
        return len(self.description_lists[0]) * len(self.description_lists)

    def __getitem__(self, index):
        nr_of_embeddings = len(self.description_lists[0])
        index_embedding = index % nr_of_embeddings
        index_description_list = math.floor(index / nr_of_embeddings)
        embedding = self.im_feats[index_embedding]
        description_vector = self.description_lists[index_description_list][index_embedding]
        return torch.FloatTensor(embedding), torch.LongTensor(description_vector)


def collate_image_captions(batch, PAD_index):    
    transposed = list(zip(*batch))

    # pad, sort and stack captions
    captions = transposed[1]
    captions_input = [c[:-1] for c in captions] # remove EOS
    caption_lengths = np.array([len(c) for c in captions_input])
    sort_indices = np.argsort(caption_lengths)[::-1].copy()
    max_length = max(caption_lengths)
    
    captions_input_padded = [
        torch.cat(
            (c, torch.LongTensor([PAD_index] * (max_length - len(c))))
        )  for c in captions_input]      
    captions_input_collated = default_collate(captions_input_padded)
    captions_input_collated_sorted = captions_input_collated[sort_indices]

    captions_target = [c[1:] for c in captions] # remove SOS
    captions_target_padded = [
        torch.cat(
            (c, torch.LongTensor([PAD_index] * (max_length - len(c))))
        )  for c in captions_target]      
    captions_target_collated = default_collate(captions_target_padded)
    captions_target_collated_sorted = captions_target_collated[sort_indices]

    # sort and stack image encodings
    im_features = transposed[0]
    image_features_collated = default_collate(im_features)
    image_features_collated_sorted = image_features_collated[sort_indices]

    return [
        image_features_collated_sorted, 
        captions_input_collated_sorted, 
        captions_target_collated_sorted, 
        torch.LongTensor(caption_lengths[sort_indices])
    ]

