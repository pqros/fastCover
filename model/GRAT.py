from model.layer.GRATLayer import GRATLayer
from model.layer.GRATVarLayer import GRATVLayer
import torch.nn as nn
import torch


# GRAT models
# List of models: GRAT2, GRAT3, GRAT4

class GRAT2_(nn.Module):  # GRAT2 before sigmoid
    def __init__(self, in_feats, hidden_feats1, out_feats, *args):
        super(GRAT2_, self).__init__()
        self.grat1 = GRATLayer(in_feats, hidden_feats1)
        self.grat2 = GRATLayer(hidden_feats1, out_feats)

    def forward(self, g, feature):
        h = torch.relu(self.grat1(g, feature))
        h = self.grat2(g, h)
        return h

class GRAT2(nn.Module):
    def __init__(self, in_feats, hidden_feats1, *args):
        super(GRAT2, self).__init__()
        self.grat = GRAT2_(in_feats, hidden_feats1, 1, *args)

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class GRAT3_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, out_feats, *args):
        super(GRAT3_, self).__init__()
        self.grat1 = GRATLayer(in_feats, hidden_feats1)
        self.grat2 = GRATLayer(hidden_feats1, hidden_feats2)
        self.grat3 = GRATLayer(hidden_feats2, out_feats)

    def forward(self, g, feature):
        h = torch.relu(self.grat1(g, feature))
        h = torch.relu(self.grat2(g, h))
        h = self.grat3(g, h)
        return h


class GRAT3(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, *args):
        super(GRAT3, self).__init__()
        self.grat = GRAT3_(in_feats, hidden_feats1, hidden_feats2, 1, *args)  # out_feats=1

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class GRAT4_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, hidden_feats3, out_feats, *args):
        super(GRAT4_, self).__init__()
        self.grat1 = GRATLayer(in_feats, hidden_feats1)
        self.grat2 = GRATLayer(hidden_feats1, hidden_feats2)
        self.grat3 = GRATLayer(hidden_feats2, hidden_feats3)
        self.grat4 = GRATLayer(hidden_feats3, out_feats)

    def forward(self, g, feature):
        h = torch.relu(self.grat1(g, feature))  # Guanhao uses tanh
        h = torch.relu(self.grat2(g, h))
        h = torch.relu(self.grat3(g, h))
        h = self.grat4(g, h)
        return h


class GRAT4(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, hidden_feats3, *args):
        super(GRAT4, self).__init__()
        self.grat = GRAT4_(in_feats, hidden_feats1, hidden_feats2, hidden_feats3, 1, *args)

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class GRATV2_(nn.Module):  # GRAT2 before sigmoid
    def __init__(self, in_feats, hidden_feats1, out_feats, *args):
        super(GRATV2_, self).__init__()
        self.grat1 = GRATVLayer(in_feats, hidden_feats1)
        self.grat2 = GRATVLayer(hidden_feats1, out_feats)

    def forward(self, g, feature):
        h = torch.relu(self.grat1(g, feature))
        h = self.grat2(g, h)
        return h

class GRATV2(nn.Module):
    def __init__(self, in_feats, hidden_feats1, *args):
        super(GRATV2, self).__init__()
        self.grat = GRATV2_(in_feats, hidden_feats1, 1, *args)

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class GRATV3_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, out_feats, *args):
        super(GRATV3_, self).__init__()
        self.grat1 = GRATVLayer(in_feats, hidden_feats1)
        self.grat2 = GRATVLayer(hidden_feats1, hidden_feats2)
        self.grat3 = GRATVLayer(hidden_feats2, out_feats)

    def forward(self, g, feature):
        h = torch.tanh(self.grat1(g, feature))  # TODO: Guanhao uses tanh
        h = torch.relu(self.grat2(g, h))
        h = self.grat3(g, h)
        return h


class GRATV3(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, *args):
        super(GRATV3, self).__init__()
        self.grat = GRATV3_(in_feats, hidden_feats1, hidden_feats2, 1, *args)  # out_feats=1

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class GRATV4_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, hidden_feats3, out_feats, *args):
        super(GRATV4_, self).__init__()
        self.grat1 = GRATVLayer(in_feats, hidden_feats1)
        self.grat2 = GRATVLayer(hidden_feats1, hidden_feats2)
        self.grat3 = GRATVLayer(hidden_feats2, hidden_feats3)
        self.grat4 = GRATVLayer(hidden_feats3, out_feats)

    def forward(self, g, feature):
        h = torch.relu(self.grat1(g, feature))  # Guanhao uses tanh
        h = torch.relu(self.grat2(g, h))
        h = torch.relu(self.grat3(g, h))
        h = self.grat4(g, h)
        return h


class GRATV4(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, hidden_feats3, *args):
        super(GRATV4, self).__init__()
        self.grat = GRATV4_(in_feats, hidden_feats1, hidden_feats2, hidden_feats3, 1, *args)

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h