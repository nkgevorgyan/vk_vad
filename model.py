import torch.nn as nn
from torch.nn import Linear, RNN, LSTM, GRU
import torch.nn.functional as F
from torch.nn.functional import softmax, relu
from torch.autograd import Variable
import torch

BATCH_SIZE = 2048
FRAMES = 3
FEATURES = 24
STEP_SIZE = 1

class VADModel(nn.Module):

    def __init__(self, features_dim, hidden_size, batch_size):
        super(VADModel, self).__init__()


        self.relu = nn.ReLU()
        self.features_dim = features_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.hidden = self.init_hidden()
        self.rnn = LSTM(input_size=self.features_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        self.lin1 = nn.Linear(self.hidden_size ** 2, 26)
        self.lin2 = nn.Linear(26, 2)

        self.softmax = nn.Softmax(dim=1)

    def init_hidden(self):
        h = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        c = Variable(torch.zeros(1, self.batch_size, self.hidden_size))

        if torch.cuda.is_available():
            h = h.cuda()
            c = c.cuda()

        return h, c

    def forward(self, x):

        x, _ = self.rnn(x, self.hidden)
        # (batch, frames, features)

        x = x.contiguous().view(-1, self.hidden_size ** 2)

        # (batch, units)

        x = self.relu(self.lin1(x))
        x = self.lin2(x)

        return self.softmax(x)

