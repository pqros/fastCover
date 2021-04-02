import dgl.function as fn
import torch
import torch.nn as nn


# GCN layer
message_func = fn.sum(src='h', out='m')
reduce_func = fn.sum(msg='m', out='h')
class DegreeGCNPlusNodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats):
        """
        :param in_feats: input dimension
        :param out_feats: output dimension
        # :param activation: activation function
        """
        super(DegreeGCNPlusNodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        
        # TODO: potential dropout
        # self.dropout = nn.Dropout(p=0.2)
        # self.activation = activation

    def forward(self, node):
        h = node.data['h']
        deg = node.data['degree'].repeat(h.size(1), 1)
        deg = torch.transpose(deg, 1, 0)
        h = h / deg
        # h = self.dropout(h)
        # h = self.activation(self.linear(h))
        h = self.linear(h)
        return {'h': h}


class DegreeGCNPlusLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        """
        :param in_feats: input dimension
        :param out_feats: output dimension
        # :param activation: activation function
        """
        super(DegreeGCNPlusLayer, self).__init__()
        self.apply_mod = DegreeGCNPlusNodeApplyModule(in_feats, out_feats)

    def forward(self, g, inputs):
        g.ndata['h'] = inputs
        g.update_all(message_func, reduce_func)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')
