from model.layer.GCNLayer import GCNLayer
from model.layer.DegreeGCNLayer import DegreeGCNLayer
import torch.nn as nn
import torch
import torch.nn.functional as F


# GCN models
# List of models:

class DGCN2_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, out_feats, *args):
        super(DGCN2_, self).__init__()
        self.gcn1 = DegreeGCNLayer(in_feats, hidden_feats1)
        self.gcn2 = DegreeGCNLayer(hidden_feats1, out_feats)

    def forward(self, g, feature):
        h = torch.relu(self.gcn1(g, feature))
        h = self.gcn2(g, h)
        return h


class DGCN2(nn.Module):
    def __init__(self, in_feats, hidden_feats1, *args):
        super(DGCN2, self).__init__()
        self.grat = DGCN2_(in_feats, hidden_feats1, 1, *args)

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class DGCN3_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, out_feats, *args):
        super(DGCN3_, self).__init__()
        self.gcn1 = DegreeGCNLayer(in_feats, hidden_feats1)
        self.gcn2 = DegreeGCNLayer(hidden_feats1, hidden_feats2)
        self.gcn3 = DegreeGCNLayer(hidden_feats2, out_feats)

    def forward(self, g, feature):
        h = torch.relu(self.gcn1(g, feature))
        h = torch.relu(self.gcn2(g, feature))
        h = self.gcn3(g, h)
        return h


class DGCN3(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, *args):
        super(DGCN3, self).__init__()
        self.grat = DGCN3_(in_feats, hidden_feats1, hidden_feats2, 1, *args)

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class DGCN4_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, hidden_feats3, out_feats, *args):
        super(DGCN4_, self).__init__()
        self.gcn1 = DegreeGCNLayer(in_feats, hidden_feats1)
        self.gcn2 = DegreeGCNLayer(hidden_feats1, hidden_feats2)
        self.gcn3 = DegreeGCNLayer(hidden_feats2, hidden_feats3)
        self.gcn4 = DegreeGCNLayer(hidden_feats3, out_feats)

    def forward(self, g, feature):
        h = torch.relu(self.gcn1(g, feature))
        h = torch.relu(self.gcn2(g, h))
        h = torch.relu(self.gcn3(g, h))
        h = self.gcn4(g, h)
        return h


class DGCN4(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, hidden_feats3, *args):
        super(DGCN4, self).__init__()
        self.grat = DGCN4_(in_feats, hidden_feats1, hidden_feats2, hidden_feats3, 1, *args)

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h
