import torch
import torch.nn as nn
import numpy as np


class KSetMaxCoverLoss(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C

    def forward(self, v, graph, *args):
        adj_mat = np.array(graph.get_adjacency().data, dtype=np.int32)
        adj_mat = torch.from_numpy(np.array(adj_mat, dtype=int)).float()
        adj_mat += torch.eye(graph.vcount())
        adj_mat = adj_mat.cuda()
        tmp = 1 - v.unsqueeze(1) * adj_mat
        tmp = torch.prod(tmp, dim=0)
        loss1 = torch.sum(tmp)
        loss2 = torch.sum(v)
        return loss1 + self.C * loss2


class KSetMaxCoverAdjLoss(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C

    def forward(self, v, adj_mat, *args):
        tmp = 1 - v.unsqueeze(1) * adj_mat.cuda()
        tmp = torch.prod(tmp, dim=0)
        loss1 = torch.sum(tmp)
        loss2 = torch.sum(v)
        return loss1 + self.C * loss2


class KSetMaxCoverLossSigmoid(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C

    def forward(self, v, graph, k):
        print(0, flush=True)
        adj_mat = np.array(graph.get_adjacency().data, dtype=np.int32)
        adj_mat = torch.from_numpy(np.array(adj_mat, dtype=int)).float().cuda()  # TODO: make it adaptive to non-cuda env
        adj_mat += torch.eye(graph.vcount())
        v = torch.sigmoid(v)
        print(1, flush=True)
        tmp = 1 - v.unsqueeze(1)*adj_mat
        print(2, flush=True)
        tmp = torch.prod(tmp, dim=0)
        loss1 = torch.sum(tmp)
        loss2 = torch.relu(torch.sum(v)-k)  # Loss with threshold (hinge loss-like)
        return loss1 + self.C * loss2


class DegreeHopfieldLoss(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C

    def forward(self, v, graph, k):
        adj_mat = torch.tensor(graph.get_adjacency().data).float()
        adj_mat += torch.eye(graph.vcount())
        tmp = 1-v.unsqueeze(1)*adj_mat
        tmp = torch.prod(tmp, dim=0)
        loss1 = torch.sum(tmp)
        d = torch.tensor(graph.degree()).float()
        loss2 = torch.sum(v * (d-k))
        return loss1 + self.C * loss2
