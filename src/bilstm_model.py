"""
bilstm_model.py
---------------
The BiLSTM neural network (our own model, built in PyTorch).

What it does: reads ONE sentence and says if it is
    negative / neutral / positive news for the company.

How it works (4 layers):

    word IDs  ->  [Embedding]  ->  [BiLSTM]  ->  [Dropout]  ->  [Linear]  ->  3 scores
                 word -> vector    reads the      prevents      scores for neg /
                                   sentence       over-fitting  neutral / pos
                                   left->right
                                   AND right->left

Why *Bi*directional?  In "Revenue did not increase", the word "not" changes the
meaning of "increase". Reading in both directions lets every word see the
words before AND after it, so the model understands context better.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 num_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()

        # 1) Embedding: turns every word ID into a learnable vector of numbers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # 2) BiLSTM: one LSTM reads forward, another reads backward
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,        # input shape = (batch, sentence_length)
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3) Dropout: randomly zeroes some values during training only
        self.dropout = nn.Dropout(dropout)

        # 4) Linear: forward memory + backward memory (2 * hidden) -> 3 class scores
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, word_ids):
        # word_ids: (batch, seq_len)
        embedded = self.embedding(word_ids)                  # (batch, seq_len, emb_dim)

        # Real length of each sentence (not counting <pad> = 0), so the LSTM
        # skips the padding instead of "reading" meaningless filler words.
        lengths = (word_ids != 0).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False)

        _, (hidden, _) = self.lstm(packed)                   # hidden: (layers*2, batch, hidden_dim)

        # last layer's final forward state + final backward state = summary of the sentence
        sentence_vector = torch.cat((hidden[-2], hidden[-1]), dim=1)   # (batch, 2*hidden_dim)

        return self.fc(self.dropout(sentence_vector))        # (batch, 3) raw scores ("logits")
