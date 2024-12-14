from typing import Optional
from torch_geometric.typing import OptTensor

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter, Linear
import torch.nn.functional as F


from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import zeros, glorot
#from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, get_laplacian
from typing import Optional

from torch_scatter import scatter_add

from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import matmul, SparseTensor
from functools import partial
from scipy.special import comb
import scipy

class polyJacobiConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, alpha: float,
                 normalization: Optional[str] = 'sym', bias: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization

        self.K = K + 1
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.basealpha = alpha

        self.alphas = torch.ones(self.K,) * float(min(1 / alpha, 1))

        #     nn.ParameterList([
        #     nn.Parameter(torch.tensor(float(min(1 / alpha, 1))),
        #                  requires_grad=False) for i in range(self.K)
        # ])

        #print('out_channnesl:{} k:{}'.format(type(out_channels), type(K)))
        # self.comb_weight = Parameter(torch.ones(1, self.K, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        # torch.nn.init.constant_(self.alpha, 0.5)
        # # torch.nn.init.constant_(self.comb_weight, 1.0)

    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None,
                 batch: OptTensor = None):
        
        adj_t = edge_index
        if isinstance(adj_t, SparseTensor):
            row, col, edge_attr = adj_t.t().coo()
            edge_index = torch.stack([row, col], dim=0)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)
        
        edge_index, edge_weight = add_self_loops(edge_index, -edge_weight,
                                                 fill_value=1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None


        N = maybe_num_nodes(edge_index)
        sparse_lap = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(N, N))

        return sparse_lap

    def forward(self, x: Tensor, edge_index, a = 1.0, b= 1.0,
                edge_weight: OptTensor = None, batch: OptTensor = None,
                lambda_max: OptTensor = None, use_norm = False):
        """"""
        adj = self.__norm__(edge_index, x.size(self.node_dim),
                                         edge_weight, self.normalization,
                                         lambda_max, dtype=x.dtype,
                                         batch=batch)

        #set alpha
        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas]
        #set a,b
        a=a
        b=b
        l=-1.0
        r=1.0

        xs = []
        xs.append(x)

        if self.K > 1:
            coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)
            coef1 *= alphas[1]
            coef2 = (a + b + 2) / (r - l)
            coef2 *= alphas[1]
            tmp = coef1 * xs[-1] + coef2 * (adj @ xs[-1])


            xs.append(tmp)

        for tmp_k in range(2, self.K):
            coef_l = 2 * tmp_k * (tmp_k + a + b) * (2 * tmp_k - 2 + a + b)
            coef_lm1_1 = (2 * tmp_k + a + b - 1) * (2 * tmp_k + a + b) * (2 * tmp_k + a + b - 2)
            coef_lm1_2 = (2 * tmp_k + a + b - 1) * (a**2 - b**2)
            coef_lm2 = 2 * (tmp_k - 1 + a) * (tmp_k - 1 + b) * (2 * tmp_k + a + b)
            tmp1 = alphas[tmp_k - 1] * (coef_lm1_1 / coef_l)
            tmp2 = alphas[tmp_k - 1] * (coef_lm1_2 / coef_l)
            tmp3 = alphas[tmp_k - 1] * alphas[tmp_k - 2] * (coef_lm2 / coef_l)
            tmp1_2 = tmp1 * (2 / (r - l))
            tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
            nx = tmp1_2 * (adj @ xs[-1]) - tmp2_2 * xs[-1]
            nx -= tmp3 * xs[-2]

            xs.append(nx)
    
        # xs = [x.unsqueeze(1) for x in xs]#[n,1,d]
        # x = torch.cat(xs, dim=1)#[n,k,d]
        # x = x * self.comb_weight#[n,k,d] [1,k,d]
        # x = torch.sum(x, dim=1)

        return xs
 


    def message(self, x_j, norm):
        print('mess')
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')


class polyChebConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, alpha: float,
                 normalization: Optional[str] = 'sym', bias: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.K = K + 1
        # self.weight = Parameter(torch.Tensor(self.K, in_channels, out_channels))



        # self.basealpha = alpha
        # self.alphas = torch.nn.ParameterList([
        #     torch.nn.Parameter(torch.tensor(float(min(1 / alpha, 1)))) for i in range(self.K -1)
        # ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.weight)
        zeros(self.bias)

    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str],
                 lambda_max, dtype: Optional[int] = None,
                 batch: OptTensor = None):

        adj_t = edge_index
        if isinstance(adj_t, SparseTensor):
            row, col, edge_attr = adj_t.t().coo()
            edge_index = torch.stack([row, col], dim=0)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        N = maybe_num_nodes(edge_index)
        edge_weight_cp = edge_weight
        edge_index_cp = edge_index
        # sparse_adj = SparseTensor.from_edge_index(edge_index, sparse_sizes = (N,N)) #to_torch_csr_tensor(edge_index, size=(N, N))#, is_coalesced = True)
        # sparse_adj = sparse_tmp.to_torch_sparse_csr_tensor()

        edge_index, edge_weight = add_self_loops(edge_index_cp, edge_weight_cp, fill_value=-1., num_nodes=num_nodes)

        assert edge_weight is not None

        sparse_lap = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(N, N))

        return sparse_lap

    def forward(self, x: Tensor, edge_index, a=1.0, b=1.0,
                edge_weight: OptTensor = None, batch: OptTensor = None,
                lambda_max: OptTensor = None, use_norm=False):
        """"""
        sparse_l = self.__norm__(edge_index, x.size(self.node_dim),
                                 edge_weight, self.normalization,
                                 lambda_max, dtype=x.dtype,
                                 batch=batch)

        # edge_index, norm = gcn_norm(
        #     edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype, add_self_loops=False)
        #
        # sparse_l = SparseTensor(row=edge_index[0], col=edge_index[1], value=norm, sparse_sizes=(x.size(0), x.size(0)))


        Tx_0 = x
        Tx_1 = x  # Dummy.

        x_list = []
        x_list.append(x)

        if self.K > 1:
            Tx_1 = self.propagate(sparse_l, x=x)

            x_list.append(Tx_1 )


        for k in range(2, self.K):
            Tx_2 = self.propagate(sparse_l, x=Tx_1)
            Tx_2 = 2. * Tx_2 - Tx_0


            x_list.append(Tx_2 )

            Tx_0, Tx_1 = Tx_1, Tx_2


        return x_list

    def message(self, x_j, norm):
        print('mess')
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, '
                f'normalization={self.normalization})')




class PolyNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(PolyNet, self).__init__()

        self.lin1 = Linear(dataset.num_features, args.hidden, bias= args.with_bias)
        self.lin2 = Linear(args.hidden, dataset.num_classes, bias= args.with_bias)

        self.base = args.base

        if args.base == 'cheb':
            self.conv_fn = polyChebConv(dataset.num_features, args.hidden, K= args.K, alpha= args.alpha)

        elif args.base == 'jacobi':
            self.conv_fn = polyJacobiConv(dataset.num_features, args.hidden, K= args.K, alpha= args.alpha)

        #elif args.base == 'bern':
        #    self.conv_fn = PolyBernConv(dataset.num_features, args.hidden, K=args.K, alpha=args.alpha)

        self.comb_weight = nn.Parameter(torch.ones((args.K + 1,))  )


        self.K = args.K
        self.dropout = args.dropout
        self.dprate = args.dprate
        self.a = args.a
        self.b = args.b
        self.alpha = args.alpha
        self.c_y = args.c_y
        self.c_f = args.c_f
        self.ret_logit = args.ret_logit


        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index


        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.conv_fn(x, edge_index, self.a, self.b)

        x = [itm.unsqueeze(1) for itm in x]  # [n,1,d]
        x = torch.cat(x, dim=1)  # [n,k,d]


        x = x * torch.reshape(self.comb_weight, (1, self.K + 1, 1) ) # [n,k,d] [1,k,1]
        x = torch.sum(x, dim=1)
        if self.ret_logit:
            return x, F.log_softmax(x, dim=1)
        return F.log_softmax(x, dim=1)


