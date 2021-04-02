import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import APPNPConv


class APPNP(nn.Module):
    def __init__(self, in_feats, hidden_feats, activations, feat_drop=0.01, edge_drop=0.01, alpha=0.1, k=10):
        """
        APP Convolution Net
        :param in_feats: input dimension
        :param hidden_feats: list of hidden dimensions' dimension
        :param activations: list of activation functions of each layer
        :param feat_drop: dropout rate
        :param edge_drop: 
        :param alpha: alpha in the formulae (alpha in the formulae)
        :param k: number of iterations (K in the formulae)
        """
        super(APPNP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, hidden_feats[0]))
        # hidden layers
        for i in range(1, len(hidden_feats)):
            self.layers.append(nn.Linear(hidden_feats[i - 1], hidden_feats[i]))

        # output layer
        # TODO: the output is fixed to be 128. Does it make sense?
        self.layers.append(nn.Linear(hidden_feats[-1], 128))
        self.activation = activations
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.lastlayer = nn.Linear(128, 1)
        # self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation[0](self.layers[0](h))
        for i in range(1, len(self.layers) - 1):
            h = self.activation[i](self.layers[i](h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(g, h)
        h = self.lastlayer(h)
        h = torch.sigmoid(h)
        return h
