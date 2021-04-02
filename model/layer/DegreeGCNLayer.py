import torch
import torch.nn as nn


# GCN layer
def message_func(edges):
    tmp = edges.src['degree'].repeat(edges.src['h'].size(1), 1)
    tmp = torch.transpose(tmp, 1, 0)
    return {'m': torch.div(edges.src['h'], torch.sqrt(tmp))}


def reduce_func(nodes):
    return {'h': torch.sum(nodes.mailbox['m'], dim=1)}


class DegreeGCNNodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats):
        """
        helper function for DegreeGCN
        :param in_feats: input dimension
        :param out_feats: output dimension
        # :param activation: activation function
        """
        super(DegreeGCNNodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

        # TODO: a potential dropout layer
        # self.dropout = nn.Dropout(p=0.2)
        # self.activation = activation

    def forward(self, node):
        # TODO: dimension of h?
        h = node.data['h']
        deg = node.data['degree'].repeat(h.size(1), 1)
        deg = torch.transpose(deg, 1, 0)
        h = h / torch.sqrt(deg)  # normalized
        h = self.linear(h)
        # h = self.activation(h)
        return {'h': h}


class DegreeGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(DegreeGCNLayer, self).__init__()
        self.apply_mod = DegreeGCNNodeApplyModule(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(message_func, reduce_func)
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata['h']
            return h
