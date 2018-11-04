import numpy as np
import torch

from models.train_predict import calculate_batch_loss
#from src.nn.train import predict
#from src.reporting.metrics import *

class MetricsCollector(object):

    def __init__(self, model, val_data, max_length, loss_criterion, device):
        self.model = model
        self.val_data = val_data
        self.loss_criterion = loss_criterion
        self.max_length = max_length
        self.device = device

        self.train_losses = []
        self.val_losses = [] 
        self.val_blue_scores = [] 
        
    def store_val_metrics(self, epoch = None, _= None):
        # store metrics on validation set
        val_loss = self.calculate_val_loss()
        val_blue = -1.0
        self.val_losses.append(val_loss)
        self.val_blue_scores.append(val_blue)

    def store_train_loss(self, _, batch_losses):
        train_loss = np.mean(batch_losses)
        self.train_losses.append(train_loss)

#    def calculate_metrics(self):
        #(log_probs, targets, _) = predict(self.model, self.val_data, self.max_length)
        #val_loss = calculate_loss(log_probs, targets, self.loss_criterion)
#        val_loss = -1.0
#        val_blue = -1.0
#        return val_loss, val_blue

    def calculate_val_loss(self):
        self.model.eval() # set in predict mode
        batch_losses = []
        with torch.no_grad():
            for batch_index, batch in enumerate(self.val_data):
                loss = calculate_batch_loss(
                    self.model, batch, self.loss_criterion, self.device
                )
                batch_losses.append(loss.item())
        return np.mean(batch_losses)
        

