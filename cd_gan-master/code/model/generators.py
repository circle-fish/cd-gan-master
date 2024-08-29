import torch.nn as nn
import tensorflow as tf


class FcGenerator(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(FcGenerator, self).__init__()
        self.layers = []

        prev_dim = input_size
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_size))
        # self.layers.append(nn.Sigmoid())
        self.layers.append(nn.ReLU(True))
        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out


class FcGeneratorReLu(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(FcGeneratorReLu, self).__init__()
        self.layers = []

        prev_dim = input_size
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_size))
        # self.layers.append(nn.Sigmoid())
        self.layers.append(nn.ReLU(True))
        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out


class FcEncoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and  TrajectoryDiscriminator"""

    def __init__(
            self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
            dropout=0.0
    ):
        super(FcEncoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        """
        Inputs: input, (h_0, c_0) 
        - input of shape (batch, input_size): tensor containing the features of the input samples.
        - h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state 
          for each element in the batch. 
          If the LSTM is bidirectional, num_directions should be 2, else it should be 1.
        - c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial 
          cell state for each element in the batch.
        Outputs: output, (h_n, c_n) 
        - output of shape (batch, output_size): tensor containing the output 
                     features (h_t) from the last layer of the NN.
        - h_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the 
          hidden state for t = seq_len.
        - c_n of shape (num_layers * num_directions, batch, hidden_size): tensor containing the 
          cell state for t = seq_len.
        Parameters:
        - embedding_dim: i.e. input_size. The number of expected features in the input x
        - h_dim:  The number of features in the hidden state h
        - num_layers:  Number of recurrent layers
        - dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, 
                with dropout probability equal to dropout. Default: 0
        """
        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        """
        Applies a linear transformation to the incoming data: y = xA^T + by=xAT+b
        Inputs:

        Outputs:

        Parameters:
        - in_features: size of each input sample
        - out_feature: size of each output sample
        """
        self.spatial_embedding = nn.Linear(2, embedding_dim)

    # what is batch ???
    #  for batch in train_loader:
    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - batch == batch_size???
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        # troch.tensor.view(): Returns a new tensor with the same data as the self tensor but of a different shape.
        # Firstly, linear embedding of input data (obs_traj)
        # to to get fixed length vectors
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        # define empty holders
        state_tuple = self.init_hidden(batch)
        # Secondly, LSTM transformation
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h