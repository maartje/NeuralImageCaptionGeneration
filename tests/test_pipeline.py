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

    #@mock.patch('builtins.print')
    @mock.patch('preprocess.read_lines', side_effect = mfs.mock_load)
    @mock.patch('torch.load', side_effect = mfs.mock_load)
    @mock.patch('torch.save', side_effect = mfs.mock_save)
    @mock.patch('tables.open_file', side_effect = mfs.mock_tables_open_file)
    def test_pipeline(self, tbl_open, save, load, read, prnt = None):
        config = TestPipeline.mfs.test_config
        filepaths = TestPipeline.mfs.filepaths
        fs = TestPipeline.mfs.mocked_file_storage
        
        # preprocess stores vocabulary and caption vectors
        preprocess(filepaths, config['preprocess'])
        self.assertTrue(filepaths['vocab'] in fs.keys())
        self.assertTrue(set(filepaths['caption_vectors_train']) < set(fs.keys()))
        self.assertTrue(set(filepaths['caption_vectors_val']) < set(fs.keys()))
        self.assertTrue(set(filepaths['caption_vectors_test']) < set(fs.keys()))

        # train stores model and epoch metrics
        # the train loss decreases with each epoch
        # the validation loss after training has decreased 
        # the BLUE score after training has increased 
        train(filepaths, config['train'])        
        metrics = fs[filepaths['epoch_metrics']]
        is_decreasing = lambda l: all(l[i] > l[i+1] for i in range(len(l)-1))
        self.assertTrue(filepaths['model'] in fs.keys())
        self.assertTrue(filepaths['epoch_metrics'] in fs.keys())        
        self.assertTrue(is_decreasing(metrics['train_losses']))
        self.assertTrue(metrics['val_losses'][0] > metrics['val_losses'][-1])
        self.assertTrue(metrics['val_blue_scores'][0] <= metrics['val_blue_scores'][-1])




