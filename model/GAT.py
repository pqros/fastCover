from model.layer.GATLayer import GATLayer
import torch.nn as nn
import torch

# GAT models
# List of models: GAT2, GAT3
class GAT2(nn.Module):
    def __init__(self, in_feats, hidden_feats1, out_feats, *args):
        super(GAT2, self).__init__()
        self.gat1 = GATLayer(in_feats, hidden_feats1)
        self.gat2 = GATLayer(hidden_feats1, 1)

    def forward(self, g, inputs):
        h = torch.relu(self.gat1(g, inputs))
        h = torch.sigmoid(self.gat2(g, h))
        return h


class GAT3_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, out_feats, *args):
        super(GAT3_, self).__init__()
        self.gat1 = GATLayer(in_feats, hidden_feats1)
        self.gat2 = GATLayer(hidden_feats1, hidden_feats2)
        self.gat3 = GATLayer(hidden_feats2, out_feats)

    def forward(self, g, inputs):
        h = torch.relu(self.gat1(g, inputs))
        h = torch.relu(self.gat2(g, h))
        h = self.gat3(g, h)
        return h


class GAT3(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, *args):
        super(GAT3, self).__init__()
        self.grat = GAT3_(in_feats, hidden_feats1, hidden_feats2, 1, *args)

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h
