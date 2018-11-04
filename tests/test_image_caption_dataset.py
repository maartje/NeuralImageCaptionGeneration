import unittest
import mock
from tests.mock_file_system import MockFileSystem
import tables
import numpy as np
from torch.utils import data

from models.image_captions_dataset import ImageCaptionsDataset, collate_image_captions

class TestImageCaptionsDataset(unittest.TestCase):
    mfs = MockFileSystem(64)

    @mock.patch('tables.open_file', side_effect = mfs.mock_tables_open_file)
    @mock.patch('torch.load', side_effect = mfs.mock_load)
    def setUp(self, load, open_file):
        TestImageCaptionsDataset.mfs.add_preprocess_results()
        filepaths = TestImageCaptionsDataset.mfs.filepaths
        self.dataset = ImageCaptionsDataset(
            filepaths['image_features_train'], 
            filepaths['caption_vectors_train']
        )
        fs = TestImageCaptionsDataset.mfs.mocked_file_storage
        self.caption_vectors = [ fs[fp] for fp in filepaths['caption_vectors_train']]
        self.im_feats = fs[filepaths['image_features_train']]
 
    def test_get_item_(self):
        self.assertEqual(self.dataset[0][1].tolist(), self.caption_vectors[0][0])
        self.assertTrue(np.allclose(self.dataset[0][0].numpy(), self.im_feats[0]))
        self.assertEqual(self.dataset[8][1].tolist(), self.caption_vectors[1][3])
        self.assertTrue(np.allclose(self.dataset[8][0].numpy(), self.im_feats[3]))

    def test_len_(self):
        self.assertEqual(len(self.dataset), 10)

    def test_collate_image_captions(self):
        pad_index = -1
        batch_size = 3
        collate_fn = lambda b: collate_image_captions(b, pad_index)
        dataloader = data.DataLoader(
            self.dataset, collate_fn = collate_fn, batch_size = batch_size
        )
        
        batch = next(iter(dataloader))
        
        im_feats, caption_in, caption_target, lengths = batch
        encoding_size = TestImageCaptionsDataset.mfs.encoding_size
        
        # check shapes
        self.assertEqual(list(im_feats.shape), [batch_size, encoding_size])
        self.assertEqual(list(caption_in.shape), [batch_size, 11])
        self.assertEqual(list(caption_target.shape), [batch_size, 11])
        
        # check reordering
        self.assertTrue(np.allclose(im_feats[0].numpy(), self.im_feats[2]))
        self.assertEqual(caption_in[0].tolist(), self.caption_vectors[0][2][:-1])
        self.assertEqual(caption_target[0].tolist(), self.caption_vectors[0][2][1:])
        self.assertEqual(lengths.tolist(), [11,8,7])

