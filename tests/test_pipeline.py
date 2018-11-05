import unittest
import mock
from tests.mock_file_system import MockFileSystem
import warnings

from preprocess import preprocess
from train import train
from predict import generate_predictions
from evaluate import evaluate

import torch
import numpy as np



class TestPipeline(unittest.TestCase):
    mfs = MockFileSystem(64, 'show_tell')
        
    def setUp(self):
        warnings.simplefilter("ignore")
        torch.manual_seed(42)
        np.random.seed(42)

    @mock.patch('builtins.print')
    @mock.patch('preprocess.read_lines', side_effect = mfs.mock_load)
    @mock.patch('predict.write_lines', side_effect = mfs.mock_save)
    @mock.patch('torch.load', side_effect = mfs.mock_load)
    @mock.patch('torch.save', side_effect = mfs.mock_save)
    @mock.patch('tables.open_file', side_effect = mfs.mock_tables_open_file)
    @mock.patch('matplotlib.pyplot.savefig')
    @mock.patch('evaluation.blue_calculator.run_multi_bleu')
    @mock.patch('evaluation.blue_calculator.run_multi_blue_compare_human_performance')
    def test_pipeline_show_tell(self, blue_human, blue, savefig, tbl_open, 
                      save, load, read, write, prnt = None):
        TestPipeline.mfs.initialize_file_storage()
        TestPipeline.mfs.update_config("show_tell")
        self.run_and_check_pipeline(blue_human, blue, savefig)

    @mock.patch('builtins.print')
    @mock.patch('preprocess.read_lines', side_effect = mfs.mock_load)
    @mock.patch('predict.write_lines', side_effect = mfs.mock_save)
    @mock.patch('torch.load', side_effect = mfs.mock_load)
    @mock.patch('torch.save', side_effect = mfs.mock_save)
    @mock.patch('tables.open_file', side_effect = mfs.mock_tables_open_file)
    @mock.patch('matplotlib.pyplot.savefig')
    @mock.patch('evaluation.blue_calculator.run_multi_bleu')
    @mock.patch('evaluation.blue_calculator.run_multi_blue_compare_human_performance')
    def tets_pipeline_show_attend_tell(self, blue_human, blue, savefig, tbl_open, 
                      save, load, read, write, prnt = None):
        TestPipeline.mfs.initialize_file_storage()
        TestPipeline.mfs.update_config("show_attend_tell")
        self.run_and_check_pipeline(blue_human, blue, savefig)
                      
    def run_and_check_pipeline(self, blue_human, blue, savefig):  
        config = TestPipeline.mfs.test_config
        filepaths = TestPipeline.mfs.filepaths
        fs = TestPipeline.mfs.mocked_file_storage
        
        # preprocess: stores vocabulary and caption vectors
        preprocess(filepaths, config['preprocess'])
        self.assertTrue(filepaths['vocab'] in fs.keys())
        self.assertTrue(set(filepaths['caption_vectors_train']) < set(fs.keys()))
        self.assertTrue(set(filepaths['caption_vectors_val']) < set(fs.keys()))
        self.assertTrue(set(filepaths['caption_vectors_test']) < set(fs.keys()))

        # train: stores model and epoch metrics
        # SANITY CHECKS:
        # - the train loss is expected to decrease with each epoch (on average)
        # - the validation loss is expected to decrease during training
        # - the validation BLUE score is expected to be reasonably high on training data
        train(filepaths, config['train'])        
        metrics = fs[filepaths['epoch_metrics']]
        is_decreasing = lambda l: all(l[i] > l[i+3] for i in range(len(l)-3))
        self.assertTrue(filepaths['model'] in fs.keys())
        self.assertTrue(filepaths['epoch_metrics'] in fs.keys())        
        self.assertTrue(is_decreasing(metrics['train_losses']))
        self.assertTrue(metrics['val_losses'][0] > np.mean(metrics['val_losses'][-3:]))
        self.assertTrue(max(metrics['val_blue_scores']) > 0.25)
        
        # predict: stores predicted sentences for test, validation and train datasets
        # SANITY CHECK:
        # - model overfits on train data 
        generate_predictions(filepaths, config['predict'])
        self.assertTrue(filepaths['predictions_train'] in fs.keys())
        self.assertTrue(filepaths['predictions_test'] in fs.keys())
        self.assertTrue(filepaths['predictions_val'] in fs.keys())
        overfits = detect_overfits(
            fs[filepaths['predictions_train']], 
            fs[filepaths['captions_train'][0]], 
            fs[filepaths['captions_train'][1]]
        )
        self.assertTrue(np.sum(overfits) > 1) # at least one exact reproduction 
        
        # evaluate:
        # - saves a plot of epoch losses
        # - saves a plot of epoch BLUE scores
        # - generates BLUE scores for test, validation and train data
        # - compares BLUE scores using the model, 
        #   with BLUE scores comparing captions created by human 
        evaluate(filepaths)
        (fname_losses, ), _ =savefig.call_args_list[0]
        (fname_blue, ), _ =savefig.call_args_list[1]
        self.assertEqual(fname_losses, filepaths['plot_epoch_loss'])
        self.assertEqual(fname_blue, filepaths['plot_epoch_bleu'])
        blue_human.assert_called_with(
            filepaths[f'captions_test'], 
            filepaths[f'predictions_test'], 
            filepaths[f'bleu_test']
        )
        blue.assert_called_with(
            filepaths[f'captions_test'], 
            filepaths[f'predictions_test'], 
            filepaths[f'bleu_test']
        )
        
        
def detect_overfits(predictions, captions_1, captions_2):
    def prediction_quality(i):
        p = predictions[i] 
        return int(p ==  captions_1[i].lower() or p == captions_2[i].lower())
    return [prediction_quality(i) for i in range(len(predictions))]
    
 



