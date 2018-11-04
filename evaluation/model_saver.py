import torch

class ModelSaver():

    def __init__(self, model, metrics_collector, fpath_model):
        self.model = model
        self.metrics_collector = metrics_collector
        self.fpath_model = fpath_model
        
    def save_best_model(self, _ = None, __ = None):
        """
        Saves model with best BLUE score on validation set. 
        Call this function after 'metricCollector.store_val_metrics'.
        """
        max_val_blue = max(self.metrics_collector.val_blue_scores) 
        last_val_blue = self.metrics_collector.val_blue_scores[-1]
        if last_val_blue == max_val_blue:
            torch.save(self.model, self.fpath_model)
            print (f"best model so far saved")
        else:
            print()
                


