import torch 
from parse_config import get_file_paths
import mock
import numpy as np

class MockFileSystem:

    def mock_glob(fpattern):
        return [fpattern]

    @mock.patch('glob.glob', side_effect = mock_glob)
    def __init__(self, encoding_size, global_feats = True):
        self.encoding_size = encoding_size
        self.mocked_file_storage = {
            'data/input/train.1.en' : [
                'Two dogs playing with a ball.',
                'A boy in black suit.',
                'Two men walking through the garden in the sun.',
                'Mountain covered with snow.',
                'A man reading the news paper.',
            ],
            'data/input/train.2.en' : [
                'Two dogs and a ball in the parc.',
                'A boy standing on the street wearing a black suit.',
                'A garden with two men smoking a sigarette.',
                'Snow on mountain top.',
                'A man reading a paper.',
            ],
            'data/input/val.1.en' : [
                'A cat lying on the roof.',
                'A man waiting at the station.',
                'A woman in a red dress.'
            ],
            'data/input/test.1.en' : [
                'A mouse eating cheese.',
                'A green parc with a fauntain',
                'A cloudy sky.'
            ]
        }
        self.mocked_file_storage['data/input/flickr30k_train_resnet50_cnn_features.hdf5'] = self.generate_random_encodings(encoding_size, 6)
        self.mocked_file_storage['data/input/flickr30k_valid_resnet50_cnn_features.hdf5'] = self.generate_random_encodings(encoding_size, 3)
        self.mocked_file_storage['data/input/flickr30k_test_resnet50_cnn_features.hdf5'] = self.generate_random_encodings(encoding_size, 3)

        self.test_config = {
            "input" : {
		        "fpattern_captions_train": "train.1.en",
		        "fpattern_captions_val": "val.1.en",
		        "fpattern_captions_test": "test.1.en",
		        "fname_image_features_train" : "flickr30k_train_resnet50_cnn_features.hdf5",
		        "fname_image_features_val" : "flickr30k_valid_resnet50_cnn_features.hdf5",
		        "fname_image_features_test" : "flickr30k_test_resnet50_cnn_features.hdf5"
            },
	        "preprocess" : {
		        "min_occurences" : 1 # do not filter rare words
	        },
	        "train" : {
	            "encoding_size" : encoding_size
	        }
        }
        self.filepaths = get_file_paths(self.test_config['input'])
        self.filepaths['captions_train'] = [
            'data/input/train.1.en', 'data/input/train.2.en'
        ]
        self.filepaths['caption_vectors_train'] = [
            'data/preprocess/train.1.en.pt', 'data/preprocess/train.2.en.pt'
        ]

    def generate_random_encodings(self, encoding_size, n):
        rgen = lambda : 2*np.random.rand(encoding_size)#.view(1,-1)
        return [rgen() for _ in range(n)]

    def mock_load(self, fpath):
        return self.mocked_file_storage[fpath]

    def mock_save(self, data, fpath):
        self.mocked_file_storage[fpath] = data
                
    def mock_tables_open_file(self, fpath, mode):
        result = mock.MagicMock()
        result.root.global_feats = self.mocked_file_storage[fpath]
        return result

    def add_preprocess_results(self):
        self.mocked_file_storage['data/preprocess/train.1.en.pt'] = [
            [0, 4, 5, 6, 7, 8, 9, 10, 1], 
            [0, 8, 11, 12, 13, 14, 10, 1], 
            [0, 4, 15, 16, 17, 18, 19, 12, 18, 20, 10, 1], 
            [0, 21, 22, 7, 23, 10, 1], 
            [0, 8, 24, 25, 18, 26, 27, 10, 1]
        ]
        self.mocked_file_storage['data/preprocess/train.2.en.pt'] = [
            [0, 4, 5, 28, 8, 9, 12, 18, 29, 10, 1], 
            [0, 8, 11, 30, 31, 18, 32, 33, 8, 13, 14, 10, 1], 
            [0, 8, 19, 7, 4, 15, 34, 8, 35, 10, 1], 
            [0, 23, 31, 21, 36, 10, 1], 
            [0, 8, 24, 25, 8, 27, 10, 1]
        ]




