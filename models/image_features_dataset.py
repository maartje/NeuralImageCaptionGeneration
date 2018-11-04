import torch
from torch.utils import data
import tables

class ImageFeaturesDataset(data.Dataset):

    def __init__(self, fpath_image_features, global_feats = True):
        super(ImageFeaturesDataset, self).__init__()
        self.fpath_image_features = fpath_image_features
        self.h5file = tables.open_file(fpath_image_features, mode="r")
        if global_feats :
            self.im_feats = self.h5file.root.global_feats
        else :
            self.im_feats = self.h5file.root.local_feats

    def __len__(self):
        return len(self.im_feats)

    def __getitem__(self, index):
        return torch.FloatTensor(self.im_feats[index])
        

