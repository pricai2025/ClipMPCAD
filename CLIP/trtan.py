import torch.nn as nn
import torch
class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, dropout):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = nn.positional_encoding(embedding_dim)
        self.transformer = nn.Transformer(embedding_dim, num_heads, num_layers, dropout)
        self.fc = nn.Linear(embedding_dim, vocab_size)
    def forward(self, x):
        # 输入x形状: (batch_size, seq_len)
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.transformer(x) # Transformer编码处理
        x = self.fc(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    lstm = LSTMModel(input_size=10, hidden_size=20, output_size=5)
