r'''LSTM model used in the dst prediction
Neural network model consisting of a single LSTM and a dense layer

    The input is a three-dimensional torch Tensor of dimensions (batches, time_backward, input_size),
    as described in the torch documentations (https://pytorch.org/docs/stable/nn.html#lstm)

    The output is a three-dimensionsal torch Tensor of dimensions (batches, 1, output_size)

author: Brecht Laperre
e-mail: brecht.laperre@kuleuven.be
'''
import torch
import torch.nn as nn


class LSTMnn(nn.Module):

    def __init__(self, input_size, num_layers, hidden_size, output_size):
        '''Model with a single LSTM module and a dense layer
        Input:
            input_size: Number of expected features for x
            num_layers: Number of layers in the LSTM
            hidden_size: Number of neurons in the hidden layer of the LSTM
            output_size: Number of timesteps in the future 
        '''
        super(LSTMnn, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden2_to_output = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def init_hidden(self, batch_size):
        return (torch.randn(self.num_layers, batch_size, self.hidden_size),
                torch.randn(self.num_layers, batch_size, self.hidden_size))

    def forward(self, x):
        """Input:
            x: Tensor of shape (batch_size, time_steps, input_size)
        Output:
            Tensor of shape (batch_size, 1, output_size)
        """
        batch_size = x.shape[0]
        hidden_state = self.init_hidden(batch_size)
        out, hidden_state = self.lstm(x, hidden_state) # out: Tensor of shape (batches, 1, hidden_size)
        # transformed to (batches, 1, output_size) using the dense layer
        return self.hidden2_to_output(out[:, -1:])
