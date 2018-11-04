import torch

class ModelSaver():

    def __init__(self, model, metrics_collector, fpath_model):
        self.model = model
        self.metrics_collector = metrics_collector
        self.fpath_model = fpath_model
        
    def save_best_model(self, _, __):
        """
        Saves model with lowest validation loss. 
        Call this function after 'metricCollector.store_val_metrics'.
        """
        min_val_loss = min(self.metrics_collector.val_losses) 
        last_val_loss = self.metrics_collector.val_losses[-1]
        if last_val_loss == min_val_loss:
            torch.save(self.model, self.fpath_model)
            print (f"best model so far saved")
        else:
            print()
                


