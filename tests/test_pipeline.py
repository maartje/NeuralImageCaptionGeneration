import unittest
import mock
import tests.mock_file_system as mfs
import warnings

from preprocess import preprocess
from parse_config import get_file_paths

class TestPipeline(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
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
        }

    @mock.patch('builtins.print')
    @mock.patch('preprocess.read_lines', side_effect = mfs.mock_load)
    @mock.patch('torch.save', side_effect = mfs.mock_save)
    @mock.patch('glob.glob', side_effect = mfs.mock_glob)
    def test_pipeline(self, glob, save, load, prnt = None):
        config = self.test_config
        filepaths = get_file_paths(config['input'])
        print(config['preprocess'])
        preprocess(filepaths, config['preprocess'])
        print (mfs.mocked_file_storage)



