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
            loss = calculate_batch_loss(model, batch, loss_criterion, device, alpha_c)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        for fn_on_epoch_completed in fn_epoch_listeners:
            fn_on_epoch_completed(epoch, batch_losses)

def predict(model, test_data, SOS_index, max_length, device):
    """ Predicts the probabilities of the target classes.
    
    Args:
        model: Image Caption Generation Model
        test_data: iterator over batches of image features
        SOS_index: index of start token
        max_length: max length of generated sentence
        device: CPU or GPU device 

    Returns:
        list of predicted index vectors representing sentences
    """
    results = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in test_data:
            image_features = batch.to(device)
            predicted_tokens = []
            batch_size = image_features.size()[0]
            image_features = image_features
            inputs = torch.LongTensor([[SOS_index]]*batch_size)
            lengths = torch.ones([batch_size], dtype=torch.long)
            hidden = None
            inputs = inputs.to(device)
            lengths = lengths.to(device)
            for i in range(max_length):
                output, hidden, *_ = model(image_features, inputs, lengths, hidden)
                _, topi = output.topk(1)
                predicted_tokens.append(topi.squeeze(1))
                inputs = topi.squeeze(1)
            result = torch.cat(predicted_tokens, 1)
            results.append(result)
        return torch.cat(results).tolist()


def calculate_batch_loss(model, batch, loss_criterion, device, alpha_c = None):
    (image_features, caption_inputs, caption_targets, caption_lengths) = batch
    image_features = image_features.to(device) 
    caption_inputs = caption_inputs.to(device)
    caption_targets = caption_targets.to(device)
    caption_lengths = caption_lengths.to(device)

    output_probs, *k = model(
        image_features, caption_inputs, caption_lengths
    )    
    loss = loss_criterion(output_probs.permute(0, 2, 1), caption_targets)
    if alpha_c and (len(k) > 1): # HACK
        # Add doubly stochastic attention regularization for show_attend_tell
        alphas = k[1]
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
    return loss

