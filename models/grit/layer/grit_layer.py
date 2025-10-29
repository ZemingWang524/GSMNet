import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add,scatter_mean

# from models.grit.utils import negate_edge_index
from torch_geometric.graphgym.register import *
import opt_einsum as oe

from yacs.config import CfgNode as CN

import warnings

import torch_geometric.nn as pygnn
import torch_geometric.graphgym.register as register
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out


class MultiHeadAttentionLayerGritSparse(nn.Module):
    """
        Proposed Attention Computation for GRIT
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 sqrt_relu=False,
                 signed_sqrt=True,
                 cfg=CN(),
                 **kwargs):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, edge_index, K_h, Q_h, E,V_h):
        src = K_h[edge_index[0]]  # (num relative) x num_heads x out_dim
        dest = Q_h[edge_index[1]]  # (num relative) x num_heads x out_dim
        score = src + dest  # element-wise multiplication

        E = E.view(-1, self.num_heads, self.out_dim * 2)
        E_w, E_b = E[:, :, :self.out_dim], E[:, :, self.out_dim:]
        # (num relative) x num_heads x out_dim
        score = score * E_w
        score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
        score = score + E_b

        score = self.act(score)
        e_t = score

        # output edge
        batch_wE = score.flatten(1)

        # final attn
        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(score, edge_index[1])  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch_attn = score

        # Aggregate with Attn-Score
        msg = V_h[edge_index[0]] * score  # (num relative) x num_heads x out_dim
        batch_wV = torch.zeros_like(V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, edge_index[1], dim=0, out=batch_wV, reduce='add')

        if self.edge_enhance and E is not None:
            rowV = scatter(e_t * score, edge_index[1], dim=0, reduce="add")
            rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
            batch_wV = batch_wV + rowV
        return batch_wV, batch_attn, batch_wE

    def forward(self, x, edge_attr, edge_index):
        Q_h = self.Q(x)
        K_h = self.K(x)

        V_h = self.V(x)

        batch_E = self.E(edge_attr)

        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        K_h = K_h.view(-1, self.num_heads, self.out_dim)
        V_h = V_h.view(-1, self.num_heads, self.out_dim)
        rbatch_wV, rbatch_attn, rbatch_wE = self.propagate_attention(edge_index, K_h, Q_h, batch_E,V_h)

        h_out = rbatch_wV
        e_out = rbatch_wE

        return h_out, e_out


@register_layer("GritTransformer")
class GritTransformerLayer(nn.Module):
    """
        Proposed Transformer Layer for GRIT
    """

    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 # cfg=dict(),
                 **kwargs):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        # -------
        self.update_e = True
        self.bn_momentum = 0.1  # BN
        self.bn_no_runner = False
        self.rezero = False

        self.act = act_dict[act]() if act is not None else nn.Identity()
        # if cfg.get("attn", None) is None:
        #     cfg.attn = dict()
        self.use_attn = True
        # self.sigmoid_deg = cfg.attn.get("sigmoid_deg", False)
        self.deg_scaler = True

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=False,
            dropout=attn_dropout,
            clamp=5.,
            act='relu',
            edge_enhance=True,
            sqrt_relu=False,
            signed_sqrt=True,
            scaled_attn=False,
            no_qk=False,
        )



        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        else:
            self.O_e = nn.Identity()
            
        # deg_drop
        self.grtidrop = nn.Dropout(0.01)
        # -------- Deg Scaler Option ------

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim // num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                momentum=0.1)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                momentum=0.1) if norm_e else nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5,
                                                momentum=0.1)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha2_h = nn.Parameter(torch.zeros(1, 1))
            self.alpha1_e = nn.Parameter(torch.zeros(1, 1))

    def forward(self, h, rrwp_edge_index_enc, rrwp_edge_attr_enc, num_nodes,log_deg):

        log_deg = get_log_deg(log_deg,num_nodes)
        
        h_in1 = h  # for first residual connection

        e_in1 = rrwp_edge_attr_enc
        e = None
        # multi-head attention out

        h_attn_out, e_attn_out = self.attention(h,rrwp_edge_attr_enc,rrwp_edge_index_enc)

        h = h_attn_out.view(num_nodes, -1)

        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)
            
        h = self.grtidrop(h)
        h = self.O_h(h)

        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = self.O_e(e)

        if self.residual:
            if self.rezero: h = h * self.alpha1_h
            h = h_in1 + h  # residual connection
            if e is not None:
                if self.rezero: e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None: e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None: e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)

        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero: h = h * self.alpha2_h
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        return h, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual,
            super().__repr__(),
        )


@torch.no_grad()
def get_log_deg(deg,num):
    deg = deg.view(num, 1)
    return deg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_scatter import scatter
import math
from torch_sparse import SparseTensor
import opt_einsum as oe

from torch_geometric.graphgym.register import act_dict
from torch_geometric.utils.num_nodes import maybe_num_nodes
from mamba_ssm import Mamba

class EWSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",dt_min=0.001, dt_max=0.1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dropout_rate = dropout

        self.edge_weight_mlp = nn.Sequential(
            nn.BatchNorm1d(d_model, momentum=0.1,eps=1e-5),
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid() 
        )

        self.mamba = CustomMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=0.001,
            dt_max=0.1,
            bias=False
        )

        self.pre_norm = nn.BatchNorm1d(d_model, momentum=0.1,eps=1e-5)
        self.output_norm = nn.BatchNorm1d(d_model, momentum=0.1,eps=1e-5)
        self.dropout = nn.Dropout(0.1)  

    def forward(self, x, edge_index, edge_attr):
        
        edge_weight = self.edge_weight_mlp(edge_attr).squeeze(-1)  # [num_edges]
        edge_weight = torch.sigmoid(edge_weight)  


        x_batch = x.unsqueeze(0)  # [1, num_nodes, d_model]
        x_out = self.mamba(x_batch,edge_index=edge_index, edge_weight=edge_weight).squeeze(0)

        x_out = self.output_norm(x_out)

        return x + x_out

class StateFormer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, attn_dropout=0.0,
                 layer_norm=False, batch_norm=True, residual=True, act='relu',
                 norm_e=True, O_e=True,
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 mamba_dt_rank="auto", mamba_dt_min=0.001, mamba_dt_max=0.1,
                 mamba_dropout=0.1):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.update_e = True
        self.bn_momentum = 0.1
        self.deg_scaler = True

        self.act = act_dict[act]() if act is not None else nn.Identity()

        self.mamba = EWSSM(
            d_model=in_dim, 
            d_state=mamba_d_state, 
            d_conv=mamba_d_conv, 
            expand=mamba_expand,
            dt_rank=mamba_dt_rank,
            dt_min=mamba_dt_min,
            dt_max=mamba_dt_max,
            dropout=mamba_dropout
        )

        self.gate = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),  
            nn.Sigmoid()    
        )

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=False,
            dropout=attn_dropout,
            clamp=5.,
            act='relu',
            edge_enhance=True
        )

        self.O_h = nn.Linear(out_dim // num_heads * num_heads, out_dim)
        self.O_e = nn.Linear(out_dim // num_heads * num_heads, out_dim) if O_e else nn.Identity()

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim // num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()
        if self.batch_norm:
            self.mamba_bn = nn.BatchNorm1d(in_dim, momentum=0.1, eps=1e-5)
            self.batch_norm1_h = nn.BatchNorm1d(out_dim, momentum=0.1,eps=1e-5)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim, momentum=0.1,eps=1e-5) if norm_e else nn.Identity()

        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim, momentum=0.1)

    def forward(self, h, rrwp_edge_index_enc, rrwp_edge_attr_enc, num_nodes, log_deg, inf_edge_index, inf_edge_features):
        h_in1 = h  # [num_nodes, in_dim]
        e_in1 = rrwp_edge_attr_enc

        h_mamba = self.mamba(h, inf_edge_index, inf_edge_features)  # [num_nodes, in_dim]

        if self.batch_norm:
            h_mamba = self.mamba_bn(h_mamba)

        gate_weight = torch.sigmoid(self.gate(torch.cat([h, h_mamba], dim=-1)))
        h = h + gate_weight * h_mamba  

        h_attn_out, e_attn_out = self.attention(h, rrwp_edge_attr_enc, rrwp_edge_index_enc)
        h = h_attn_out.view(num_nodes, -1)

        if self.deg_scaler:
            log_deg = get_log_deg(log_deg, num_nodes)
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.O_h(h)
        e = self.O_e(e_attn_out.flatten(1)) if e_attn_out is not None else None

        if self.residual:
            h = h_in1 + h
            if e is not None:
                e = e_in1 + e

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None:
                e = self.layer_norm1_e(e)
        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        h_in2 = h
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)
        if self.batch_norm:
            h = self.batch_norm2_h(h)

        return h, e
    

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from torch_scatter import scatter


class CustomMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, edge_index, edge_weight):
        batch, seqlen, dim = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        x, z = xz.chunk(2, dim=1)

        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen)
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen)

        if edge_index is not None and edge_weight is not None:
            edge_weight = edge_weight.view(1, 1, -1)  # [1, 1, num_edges]
            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            B_weighted = torch.zeros_like(B)  # [batch, d_state, seqlen]
            C_weighted = torch.zeros_like(C)
            scatter(edge_weight * B[:, :, src_nodes], dst_nodes, dim=-1, out=B_weighted, reduce='sum')
            scatter(edge_weight * C[:, :, src_nodes], dst_nodes, dim=-1, out=C_weighted, reduce='sum')
            B = B + B_weighted
            C = C + C_weighted

        y = selective_scan_fn(
            x,
            dt,
            -torch.exp(self.A_log.float()),
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out