import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm as w_norm


class Decoder(nn.Module):
    def __init__(
        self,
        args,
        dropout_prob=0.1,
    ):
        super(Decoder, self).__init__()

        # **** YOU SHOULD IMPLEMENT THE MODEL ARCHITECTURE HERE ****
        # Define the network architecture based on the figure shown in the assignment page.
        # Read the instruction carefully for layer details.
        # Pay attention that your implementation should include FC layers, weight_norm layers,
        # PReLU layers, Dropout layers and a tanh layer.
        self.dropout_prob = dropout_prob
        self.args = args

        self.h1  = w_norm(nn.Linear(3, 512))     
        self.h2  = w_norm(nn.Linear(512, 512))
        self.h3  = w_norm(nn.Linear(512, 512))

        self.h4  = w_norm(nn.Linear(512, 509))

        self.h5  = w_norm(nn.Linear(512, 512))
        self.h6  = w_norm(nn.Linear(512, 512))
        self.h7  = w_norm(nn.Linear(512, 512))
        self.h8  = nn.Linear(512, 1)

        self.prelu_layer = nn.PReLU()
        self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        self.th = nn.Tanh()
        # ***********************************************************************

    # input: N x 3
    def forward(self, input):

        # **** YOU SHOULD IMPLEMENT THE FORWARD PASS HERE ****
        # Based on the architecture defined above, implement the feed forward procedure
        x_in = input

        x = self.dropout_layer(self.prelu_layer(self.h1(input)))
        x = self.dropout_layer(self.prelu_layer(self.h2(x)))
        x = self.dropout_layer(self.prelu_layer(self.h3(x)))
        x = self.dropout_layer(self.prelu_layer(self.h4(x)))

        x = torch.hstack((x, x_in))

        x = self.dropout_layer(self.prelu_layer(self.h5(x)))
        x = self.dropout_layer(self.prelu_layer(self.h6(x)))
        x = self.dropout_layer(self.prelu_layer(self.h7(x)))

        x = self.th(self.h8(x))
        # ***********************************************************************


        return x
    
