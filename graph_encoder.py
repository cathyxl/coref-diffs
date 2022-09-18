import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import math


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
            x [bs, seq_l, d]
            y [bs, seq_l, d]
        """

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class BatchRGATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, e_in_features, out_features, dropout, alpha, concat=True, adj_thre=0.0):
        super(BatchRGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.e_in_features = e_in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj_thre = adj_thre

        self.W = Parameter(torch.zeros(size=(in_features, out_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        self.W1 = Parameter(torch.zeros(size=(e_in_features, out_features)), requires_grad=True)
        nn.init.xavier_uniform_(self.W1, gain=1.414)
        self.a = Parameter(torch.zeros(size=(3*out_features, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, edge, adj):
        """
            x: (b, n_k, in_features)
            edge: (b, n_k, n_k, e_in_features)
            adj: (b, n_k, n_k)
        """
        b, n_k, input_h = x.size()
        h = torch.matmul(x, self.W) # (b, n_k, out_features)
        e_h = torch.matmul(edge, self.W1) # (b, n_k, n_k, out_features)

        a_input = torch.cat([h.repeat(1, 1, n_k).view(b, n_k * n_k, -1), h.repeat(1, n_k, 1)], dim=2).view(b, n_k, n_k, 2 * self.out_features) # (b, n_k, n_k, out_features*2)
        # add edge hidden to calculate attention
        a_input = torch.cat([a_input, e_h], dim=-1) # (b, n_k, n_k, out_features*3)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3)) # (b, n_k, n_k)

        zero_vec = (-float('inf'))*torch.ones_like(e)
        attention = torch.where(adj > self.adj_thre, e, zero_vec) # (b, n_k, n_k)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h) # (b, n_k, out_features)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class RGATEncoderLayer(nn.Module):
    def __init__(self, nfeat, efeat, nhid, dropout, alpha, nheads, ofeat=0, adj_thre=0.0):
        """Dense version of RGATEncoderLayer."""
        super(RGATEncoderLayer, self).__init__()
        self.dropout = dropout
        if ofeat == 0:
            ofeat = nfeat
        att_hid = ofeat // nheads
        self.attentions = nn.ModuleList(
            [BatchRGATLayer(nfeat, efeat, att_hid, dropout=dropout, alpha=alpha, concat=True, adj_thre=adj_thre) for _ in range(nheads)]
        )

        self.out_trans = nn.Sequential(
            nn.Linear(att_hid*nheads, ofeat),
            nn.ReLU(inplace=True)
        )

        self.feed_forward = PositionwiseFeedForward(ofeat, nhid, 0.1)

    def forward(self, x, edge, adj):
        """
            x: (b, n_k, d)
            edge: (b, n_k, n_k, d)
            adj: (b, n_k, n_k)
        out:
            (b, n_k, d)
        """
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, edge, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.out_trans(x)
                
        out = self.feed_forward(x)

        return out
