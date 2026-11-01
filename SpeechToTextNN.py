#!/usr/bin/python3

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SpeechToTextNN(nn.Module):
    def __init__(self, n_mels, d, n_cls):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU())
        f = ((n_mels - 1) // 2) // 2 + 1
        self.proj = nn.Linear(32 * f, d)
        self.rnn = nn.LSTM(d, d, num_layers=3, bidirectional=True, batch_first=True, dropout=0.2)
        self.out = nn.Linear(2 * d, n_cls)

    def forward(self, x, x_len):
        x = self.conv(x.unsqueeze(1))
        x = self.proj(x.permute(0, 2, 1, 3).flatten(2))
        T = x.shape[1]
        x_len = x_len.clamp(max=T)
        packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(packed)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=T)
        return self.out(x).log_softmax(-1)
