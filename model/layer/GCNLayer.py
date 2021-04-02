import dgl.function as fn
import torch.nn as nn


# https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
message_func = fn.copy_src(src='h', out='m')
reduce_func = fn.sum(msg='m', out='h')
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(message_func, reduce_func)
            h = g.ndata['h']
            return self.activation(self.linear(h))
