import unittest
import mock
from tests.mock_file_system import MockFileSystem
import warnings

from preprocess import preprocess
from train import train



class TestPipeline(unittest.TestCase):
    mfs = MockFileSystem(64)
        
    def setUp(self):
        warnings.simplefilter("ignore")

    @mock.patch('builtins.print')
    @mock.patch('preprocess.read_lines', side_effect = mfs.mock_load)
    @mock.patch('torch.load', side_effect = mfs.mock_load)
    @mock.patch('torch.save', side_effect = mfs.mock_save)
    @mock.patch('tables.open_file', side_effect = mfs.mock_tables_open_file)
    def test_pipeline(self, tbl_open, save, load, read, prnt = None):
        config = TestPipeline.mfs.test_config
        filepaths = TestPipeline.mfs.filepaths
        preprocess(filepaths, config['preprocess'])
        train(filepaths, config['train'])



