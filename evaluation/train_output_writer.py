import datetime

class TrainOutputWriter(object):
    def __init__(self, metricCollector):
        self.metricCollector = metricCollector

    def print_epoch_info(self, epoch, _):
        """
        Print train and validation loss after epoch. 
        Call this function after 'metricCollector.store_train_loss'
        and 'metricCollector.store_val_metrics'.
        """
        if epoch == 0:
            initial_val_loss = self.metricCollector.val_losses[0]
            initial_val_blue = self.metricCollector.val_blue_scores[0]
            print()
            print('initial val-loss:', f'{initial_val_loss:0.3}',
                  '\t\t initial val-blue:', f'{initial_val_blue:0.3}')
            print('epoch \t train loss \t val loss \t val BLEU \t time')
        train_loss = self.metricCollector.train_losses[-1]
        val_loss = self.metricCollector.val_losses[-1]
        val_blue = self.metricCollector.val_blue_scores[-1]
        print(epoch, 
              '\t', f'{train_loss:0.3}', 
              '\t\t', f'{val_loss:0.3}', 
              '\t\t', f'{val_blue:0.3}',
              '\t\t', datetime.datetime.now().time().isoformat(timespec='seconds'),
              end='\t')
