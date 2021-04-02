import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax


# GRAT more similar to GAT 
class GRATVLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GRATVLayer, self).__init__()
        # linear layer
        self.fc = nn.Linear(in_feats, out_feats, bias=False)  # bias=True in Guanhao's original code
        # attention layer
        self.attn_fc = nn.Linear(2 * out_feats, 1, bias=False)  # bias=True in Guanhao's original code
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'h': edges.src['z'] * edges.data['alpha']}  # message divided by weight

    def reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['h'], dim=1)}

    def forward(self, g, feature):
        with g.local_scope():
            z = self.fc(feature)
            g.ndata['z'] = z
            # Equation (2)
            g.apply_edges(self.edge_attention)  # calculate e_{ij}
            # Calculate softmax on source code -> on the reversed graph
            rg = g.reverse(copy_ndata=False, copy_edata=True)
            g.edata['alpha'] = edge_softmax(rg, rg.edata['e'])
            # Equation (3)
            g.update_all(self.message_func, self.reduce_func)
            # output            
            h = g.ndata['h']
            return h
