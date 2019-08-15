import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    '''
    A simple 2-layer bidirectional LSTM classifier for unimodal language data
    '''
    def __init__(self, emb_dim, vocab_size, hidden_dim, fc_dim, output_dim, dropout_rate):
        super(LSTMClassifier, self).__init__()
        # record settings
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # construct modules
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.rnn1 = nn.LSTM(emb_dim, hidden_dim, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * hidden_dim, hidden_dim, bidirectional=True)
        self.fc1 = nn.Linear(4 * hidden_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm((hidden_dim * 2,))

    def forward(self, sentences, lengths):
        batch_size = lengths.size(0)
        setences = self.embed(sentences)
        packed_sentences = pack_padded_sequence(sentences, lengths)
        packed_h1, (final_h1, _) = self.rnn1(packed_sentences)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = self.layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)
        _, (final_h2, _) = self.rnn2(packed_normed_h1)

        # use the rnn states to predict the output
        h = torch.cat((final_h1, final_h2), dim=0).permute(1, 0, 2).contiguous().view(batch_size, -1)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o

