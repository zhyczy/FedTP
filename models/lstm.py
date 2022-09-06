import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


class TwoLinearLayers(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, bias=False):
        super(TwoLinearLayers, self).__init__()
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.num_classes = output_dimension

        self.fc1 = nn.Linear(input_dimension, hidden_dimension, bias=bias)
        self.fc2 = nn.Linear(hidden_dimension, output_dimension, bias=bias)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)


    def produce_feature(self, input_):
        self.rnn.flatten_parameters()

        encoded = self.encoder(input_)
        output, (hidden, cell) = self.rnn(encoded)
        return output


    def forward(self, input_):
        self.rnn.flatten_parameters()

        encoded = self.encoder(input_)
        output, (hidden, cell) = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)

        hidden = hidden.permute(1, 0, 2)  # change to (B, N_LAYERS, T)
        cell = cell.permute(1, 0, 2)  # change to (B, N_LAYERS, T)

        return output, (hidden, cell)