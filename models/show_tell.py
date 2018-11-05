import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, drop_out):
        super(Encoder, self).__init__()
        self.ll = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p = drop_out)

    def forward(self, im_features):
        features = im_features.unsqueeze(0)
        h_0 = self.dropout(F.relu(self.ll(features)))
        c_0 = self.dropout(F.relu(self.ll(features)))
        return (h_0, c_0)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, pad_index, drop_out):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pad_index = pad_index

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        
        self.dropout_embedding = nn.Dropout(p = drop_out)
        self.dropout_lstm = nn.Dropout(p = drop_out)

    def forward(self, hidden, input_data, seq_lengths):
        output = self.dropout_embedding(self.embedding(input_data))
        packed = pack_padded_sequence (
            output, seq_lengths, batch_first=True)
        output, hidden = self.lstm(packed, hidden)
        unpacked = pad_packed_sequence(
            output, batch_first=True, padding_value=self.pad_index, total_length=None)
        output = self.out(self.dropout_lstm(unpacked[0]))
        output = self.logsoftmax(output)
        return output, hidden
        
class ShowTell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, pad_index, 
                 drop_out = 0.3, device = None):
        super(ShowTell, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, drop_out)
        self.decoder = Decoder(hidden_size, output_size, pad_index, drop_out)
    
    def forward(self, im_features, input_data, seq_lengths, state = None, device=None):
        if state is None:
            state = self.encoder(im_features)
        return self.decoder(state, input_data, seq_lengths) 

