import unittest
import mock
from tests.mock_file_system import MockFileSystem
import warnings

from preprocess import preprocess



class TestPipeline(unittest.TestCase):
    mfs = MockFileSystem(64)
        
    def setUp(self):
        warnings.simplefilter("ignore")

    @mock.patch('builtins.print')
    @mock.patch('preprocess.read_lines', side_effect = mfs.mock_load)
    @mock.patch('torch.save', side_effect = mfs.mock_save)
    def test_pipeline(self, save, load, prnt = None):
        config = TestPipeline.mfs.test_config
        filepaths = TestPipeline.mfs.filepaths
        #print(config['preprocess'])
        #print(filepaths)
        preprocess(filepaths, config['preprocess'])
        print()
        print()
        print (TestPipeline.mfs.mocked_file_storage[filepaths['caption_vectors_train'][0]])
        print(filepaths['caption_vectors_train'][0])
        print()



