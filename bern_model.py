from typing import Optional
from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
from torch_sparse import matmul, SparseTensor

class Bern_prop(MessagePassing):
    def __init__(self, K, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)
        
        self.K = K
        self.temp = Parameter(torch.Tensor(self.K+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1)

    def forward(self, x, edge_index,edge_weight=None):
        TEMP=F.relu(self.temp)

        N = x.size(0)
        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))
        sparse_lap1 = SparseTensor(row=edge_index1[0], col=edge_index1[1], value=norm1, sparse_sizes=(N, N))

        #2I-L
        edge_index2, norm2=add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))
        sparse_lap2 = SparseTensor(row=edge_index2[0], col=edge_index2[1], value=norm2, sparse_sizes=(N, N))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
            #x=self.propagate(edge_index2,x=x,norm=norm2,size=None)
            x = self.propagate(sparse_lap2, x=x)
            tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
            x=tmp[self.K-i-1]
            #x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
            x = self.propagate(sparse_lap1, x=x)
            for j in range(i):
                #x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
                x = self.propagate(sparse_lap1, x=x)

            out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x
        return out
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
            return matmul(adj_t, x, reduce=self.aggr)


    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class BernNet(torch.nn.Module):
    def __init__(self,dataset, args):
        super(BernNet, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = Bern_prop(args.K)

        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        #x= self.m(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
