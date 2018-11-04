import torch

def fit(model, train_data, loss_criterion, optimizer, 
        epochs, device, fn_epoch_listeners = [], alpha_c = 1.0):
    """Fits the model on the training data.
    
    Args:
        model: Image Caption Generation Model
        train_data: iterator over batches
        loss_criterion: nn.NLLLoss
        optimizer: optimizer
        epochs: number of epochs used to train the model
        device: CPU or GPU device
        fn_epoch_listeners: list of functions that are called after each epoch
        alpha_c: regularization term (only used in attention model)
    """
    for epoch in range(epochs):
        model.train() # set in train mode
        batch_losses = []
        for batch_index, batch in enumerate(train_data):
            optimizer.zero_grad()
            (image_features, caption_inputs, caption_targets, caption_lengths) = batch
            image_features = image_features.to(device) 
            caption_inputs = caption_inputs.to(device)
            caption_targets = caption_targets.to(device)
            caption_lengths = caption_lengths.to(device)

            output_probs, *k = model(
                image_features, caption_inputs, caption_lengths
            )    
            loss = loss_criterion(output_probs.permute(0, 2, 1), caption_targets)
            if len(k) > 1: # HACK
                # Add doubly stochastic attention regularization for show_attend_tell
                alphas = k[1]
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        for fn_on_epoch_completed in fn_epoch_listeners:
            fn_on_epoch_completed(epoch, batch_losses)
        print(sum(batch_losses))
            

def predict(model, test_data, max_length, model_name='rnn', device=torch.device('cpu')):
    """ Predicts the probabilities of the target classes.
    
    Args:
        model: Language Identification Model
        test_data: iterator over batches of testdata

    Returns:
        log probabilities [BatchSize x Max-SequenceLength x NrOfTargetClasses]
        targets [BatchSize x Max-SequenceLength] (0 is used for padding)
        lengths [BatchSize]
    """
    model.eval() # set in predict mode

