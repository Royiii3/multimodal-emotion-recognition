"""
Standalone text emotion classifier
Embedding -> BiLSTM -> FC -> 7 emotions
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEmotionClassifier(nn.Module):
    """Pure text emotion recognition"""

    def __init__(self, vocab_size: int = MAX_VOCAB_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, TEXT_EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=TEXT_EMBED_DIM,
            hidden_size=TEXT_LSTM_HIDDEN,
            num_layers=TEXT_LSTM_LAYERS,
            dropout=TEXT_LSTM_DROPOUT if TEXT_LSTM_LAYERS > 1 else 0,
            bidirectional=TEXT_LSTM_BIDIRECTIONAL,
            batch_first=True,
        )
        lstm_dim = TEXT_LSTM_HIDDEN * 2 if TEXT_LSTM_BIDIRECTIONAL else TEXT_LSTM_HIDDEN
        self.dropout = nn.Dropout(TEXT_LSTM_DROPOUT)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, max_len] token ids
        returns: [B, NUM_CLASSES] logits
        """
        emb = self.embedding(x)                     # [B, L, E]
        emb = self.dropout(emb)
        lstm_out, (h_n, _) = self.lstm(emb)         # [B, L, H*2]
        if TEXT_LSTM_BIDIRECTIONAL:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]
        return self.classifier(h)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
