import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal
from torch_geometric.nn import Linear, MessagePassing, global_mean_pool
from torch_geometric.nn.models.schnet import ShiftedSoftplus

from models.base import BaseSettings

from models.utils import RBFExpansion

from torch_sparse import SparseTensor
import torch_geometric.transforms as T

from grit.encoder.rrwp_encoder import RRWPLinearNodeEncoder, RRWPLinearEdgeEncoder, full_edge_index
from grit.layer.grit_layer import GritTransformerLayer, MultiHeadAttentionLayerGritSparse, pyg_softmax, StateFormer


#

class GSMNetConfig(BaseSettings):
    name: Literal["gsmnet"]
    conv_layers: int = 3
    atom_input_features: int = 92
    inf_edge_features: int = 64
    fc_features: int = 256
    output_dim: int = 256
    output_features: int = 1
    rbf_min = -4.0
    rbf_max = 4.0
    potentials = []
    euclidean = False
    charge_map = False
    transformer = False
    ssm = False
    ksteps: int = 16

    # 新增 Mamba SSM 参数
    mamba_d_state: int = 64
    mamba_d_conv: int = 2
    mamba_expand: int = 2
    mamba_dt_rank: str = "auto"  # 可以是"auto"或整数
    mamba_dt_min: float = 0.001
    mamba_dt_max: float = 0.1
    mamba_dropout: float = 0.1

    class Config:
        """Configure model settings behavior."""
        env_prefix = "jv_model"


class PotNetConv(MessagePassing):

    def __init__(self, fc_features, cutoff_upper=5.0):
        super(PotNetConv, self).__init__(node_dim=0)
        self.bn = nn.BatchNorm1d(fc_features)
        self.bn_interaction = nn.BatchNorm1d(fc_features)
        self.nonlinear_full = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features)
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(3 * fc_features, fc_features),
            nn.SiLU(),
            nn.Linear(fc_features, fc_features),
        )
        self.cutoff = DistanceAttenuation(0.0, cutoff_upper)

    def forward(self, x, edge_index, edge_attr, edge_length):
        out = self.propagate(
            edge_index, x=x, edge_attr=edge_attr, edge_length=edge_length, size=(x.size(0), x.size(0))
        )
        return F.relu(x + self.bn(out))

    def message(self, x_i, x_j, edge_attr, edge_length, index):
        envelope = self.cutoff(edge_length).unsqueeze(-1)    # [num_edges, 1]
        # score = torch.sigmoid(self.bn_interaction(self.nonlinear_full(torch.cat((x_i, x_j, edge_attr), dim=1))))
        # msg = score * self.nonlinear(torch.cat((x_i, x_j, edge_attr), dim=1))
        combined = torch.cat((x_i, x_j, edge_attr), dim=1)
        score = torch.sigmoid(self.bn_interaction(self.nonlinear_full(combined)))
        msg = score * self.nonlinear(combined)
        return envelope * msg


class GSMNet(nn.Module):

    def __init__(self, config: GSMNetConfig = GSMNetConfig(name="gsmnet")):
        super().__init__()
        self.config = config

        # embedding
        if not config.charge_map:
            self.atom_embedding = nn.Linear(
                config.atom_input_features, config.fc_features
            )
        else:
            self.atom_embedding = nn.Linear(
                config.atom_input_features + 10, config.fc_features
            )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=config.rbf_min,
                vmax=config.rbf_max,
                bins=config.fc_features,
            ),
            nn.Linear(config.fc_features, config.fc_features),
            nn.SiLU(),
        )

        # 添加角度特征的RBF层
        self.rbf_angle = nn.Sequential(
            RBFExpansion(
                vmin=-1.0,
                vmax=1.0,
                bins=config.fc_features,
            ),
            nn.Linear(config.fc_features, config.fc_features),
            nn.Softplus(),
        )

        if not self.config.euclidean:
            self.inf_edge_embedding = RBFExpansion(
                vmin=config.rbf_min,
                vmax=config.rbf_max,
                bins=config.inf_edge_features,
                type='multiquadric'
            )

            self.infinite_linear = nn.Linear(config.inf_edge_features, config.fc_features)

            self.infinite_bn = nn.BatchNorm1d(config.fc_features)

        # 添加边特征更新层
        self.edge_update_layer = EdgeUpdateLayer(
            in_channels=config.fc_features,
            out_channels=config.fc_features,
            
        )

        # local module potnet
        self.local_modules = nn.ModuleList(
            [
                PotNetConv(config.fc_features)
                for _ in range(config.conv_layers)
            ]
        )

        # grit embedding
        self.rrwp_rel_encoder = RRWPLinearEdgeEncoder(config.ksteps, config.fc_features,
                                                      pad_to_full_graph=True,
                                                      add_node_attr_as_self_loop=False,
                                                      fill_value=0.)

        # global module grit
        if config.ssm:
            self.global_modules = nn.ModuleList(
                [
                    StateFormer(
                        in_dim=config.fc_features,
                        out_dim=config.fc_features,
                        num_heads=8,
                        dropout=0.,
                        act='relu',
                        attn_dropout=0.1,
                        layer_norm=False,
                        batch_norm=True,
                        residual=True,
                        norm_e=True,
                        O_e=True,
                        # 添加 Mamba 参数
                        mamba_d_state=config.mamba_d_state,
                        mamba_d_conv=config.mamba_d_conv,
                        mamba_expand=config.mamba_expand,
                        mamba_dt_rank=config.mamba_dt_rank,
                        mamba_dt_min=config.mamba_dt_min,
                        mamba_dt_max=config.mamba_dt_max,
                        mamba_dropout=config.mamba_dropout)
                    for _ in range(config.conv_layers)
                ]
            )

        elif not config.euclidean and config.transformer:
            self.global_modules = nn.ModuleList(
                [
                    GritTransformerLayer(
                        in_dim=config.fc_features,
                        out_dim=config.fc_features,
                        num_heads=8,
                        dropout=0.,
                        act='relu',
                        attn_dropout=0.1,
                        layer_norm=False,
                        batch_norm=True,
                        residual=True,
                        norm_e=True,
                        O_e=True,)
                    for _ in range(config.conv_layers)
                ]
            )

        # FC layer
        self.fc = nn.Sequential(
            nn.Linear(config.fc_features, config.fc_features), ShiftedSoftplus()
        )

        self.fc_out = nn.Linear(config.output_dim, config.output_features)

        self.edge_dir_mlp = nn.Sequential(
            nn.Linear(config.fc_features + 3, config.fc_features),
            nn.SiLU(),
            nn.Linear(config.fc_features, config.fc_features)
        )


    def forward(self, data, print_data=False):
        """CGCNN function mapping graph to outputs."""
        # fixed edge features: RBF-expanded bondlengths
        edge_index = data.edge_index
        edge_dir = data.edge_attr  # [num_edges, 3]
        edge_length = torch.norm(data.edge_attr, dim=-1)
        if self.config.euclidean:
            edge_features = self.edge_embedding(edge_length)
        else:
            edge_features = self.edge_embedding(-0.75 / edge_length)

        # 拼接方向信息
        edge_features = torch.cat([edge_features, edge_dir], dim=-1)  # [num_edges, fc_features+3]
        edge_features = self.edge_dir_mlp(edge_features)  # 新增MLP层

        # 计算边的邻居长度和角度
        edge_nei_len = None
        edge_nei_angle = None
        if hasattr(data, 'edge_nei') and data.edge_nei is not None:
            edge_nei_len = -0.75 / torch.norm(data.edge_nei, dim=-1)  # [num_edges, 3]
            edge_nei_angle = bond_cosine(data.edge_attr, data.edge_nei)  # [num_edges, 3]
            num_edge = edge_features.shape[0]
            # 3. RBF特征转换
            edge_nei_len = self.edge_embedding(edge_nei_len.reshape(-1)).reshape(num_edge, 3, -1)
            edge_nei_angle = self.rbf_angle(edge_nei_angle.reshape(-1)).reshape(num_edge, 3, -1)

        # process inf
        if not self.config.euclidean:
            inf_edge_index = data.inf_edge_index
            inf_feat = sum([data.inf_edge_attr[:, i] * pot for i, pot in enumerate(self.config.potentials)])
            inf_edge_features = self.inf_edge_embedding(inf_feat)
            inf_edge_features = self.infinite_bn(F.softplus(self.infinite_linear(inf_edge_features)))


        # initial node features: atom feature network...
        if self.config.charge_map:
            node_features = self.atom_embedding(torch.cat([data.x, data.g_feats], -1))
        else:
            node_features = self.atom_embedding(data.x)

        if not self.config.euclidean and not self.config.transformer:
            edge_index = torch.cat([data.edge_index, inf_edge_index], 1)
            edge_features = torch.cat([edge_features, inf_edge_features], 0)

        # grit_parameters
        node_nums = data.x.size(0)
        log_deg = data.log_deg
        rrwp_index_enc = data.rrwp_index
        rrwp_val_enc = data.rrwp_val
        rrwp_edge_index_enc, rrwp_edge_attr_enc = self.rrwp_rel_encoder(rrwp_index_enc, rrwp_val_enc, inf_edge_index,
                                                                        inf_edge_features, node_nums)
        edge_features = self.edge_update_layer(edge_features, edge_nei_len, edge_nei_angle)

        for i in range(self.config.conv_layers):
            if not self.config.euclidean and self.config.transformer:
                local_node_features = self.local_modules[i](node_features, edge_index, edge_features, edge_length)

                if self.config.ssm:
                    node_features, rrwp_edge_attr_enc = self.global_modules[i](
                        node_features, rrwp_edge_index_enc, rrwp_edge_attr_enc,
                        node_nums, log_deg, inf_edge_index, inf_edge_features,
                        data.batch,
                        getattr(data, "ptr", None),
                        getattr(data, "mamba_order", None),
                        getattr(data, "mamba_inv", None),
                    )
                else:
                    node_features, rrwp_edge_attr_enc = self.global_modules[i](node_features, rrwp_edge_index_enc,
                                                                                        rrwp_edge_attr_enc, node_nums,
                                                                                        log_deg)

                node_features = local_node_features + node_features

            else:
                node_features = self.local_modules[i](node_features, edge_index, edge_features)

        features = global_mean_pool(node_features, data.batch)

        features = self.fc(features)
        return torch.squeeze(self.fc_out(features))

        # pred = torch.squeeze(self.fc_out(features))
        # return pred, features

# def bond_cosine(r1, r2):
#     # r1: [num_edges, 3], r2: [num_edges, 3, 3]
#     # 计算每条边与其3个相邻边的夹角余弦
#     r1_expand = r1.unsqueeze(1).expand(-1, 3, -1)  # [num_edges, 3, 3]
#     cos = torch.sum(r1_expand * r2, dim=-1) / (
#         torch.norm(r1_expand, dim=-1) * torch.norm(r2, dim=-1) + 1e-8
#     )
#     return torch.clamp(cos, -1, 1)  # [num_edges, 3]

def bond_cosine(r1, r2):
    # 优化向量化计算
    r1_expand = r1.unsqueeze(1)  # [num_edges, 1, 3]
    # 使用einsum替代多次展开和乘法操作
    dot_product = torch.einsum('eij,eij->ei', r1_expand, r2)
    norm_r1 = torch.norm(r1, dim=-1, keepdim=True)
    norm_r2 = torch.norm(r2, dim=-1)
    cos = dot_product / (norm_r1 * norm_r2 + 1e-8)
    return torch.clamp(cos, -1, 1)

# ...existing code...
class EdgeUpdateLayer(nn.Module):
    """简化的边特征更新层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        
        # 边特征变换
        self.lin_edge = nn.Linear(in_channels, out_channels)
        
        # 邻居边长度特征处理
        self.lin_nei_len = nn.Linear(in_channels, out_channels)
        
        # 角度特征处理
        self.lin_angle = nn.Linear(in_channels, out_channels)
        
        # 特征融合网络 - 简化版本
        self.edge_update = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # 单一的归一化层
        self.norm = nn.LayerNorm(out_channels)
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )
    
    def forward(self, edge_features, edge_nei_len, edge_nei_angle):
        """
        edge_features: [num_edges, hidden_dim]
        edge_nei_len: [num_edges, 3, hidden_dim] - 已经过RBF转换的邻居边长度特征
        edge_nei_angle: [num_edges, 3, hidden_dim] - 已经过RBF转换的邻居边夹角特征
        """
        # # 变换原始边特征
        # edge_feat = self.lin_edge(edge_features)
        
        # # 处理邻居边特征和角度特征并平均聚合
        # len_feat = self.lin_nei_len(edge_nei_len.reshape(-1, self.in_channels)).reshape(-1, 3, self.out_channels)
        # angle_feat = self.lin_angle(edge_nei_angle.reshape(-1, self.in_channels)).reshape(-1, 3, self.out_channels)
        
        # # 聚合邻居信息
        # len_feat = len_feat.mean(dim=1)   # [num_edges, out_channels]
        # angle_feat = angle_feat.mean(dim=1)  # [num_edges, out_channels]
        
        # # 计算更新信息
        # update = self.edge_update(torch.cat([edge_feat, len_feat, angle_feat], dim=-1))

        # 减少重塑操作，预先计算批次大小
        batch_size = edge_features.shape[0]
        
        # 直接在原始维度上操作，避免重复reshape
        edge_feat = self.lin_edge(edge_features)
        
        # 批量处理邻居特征
        if edge_nei_len is not None and edge_nei_angle is not None:
            # 使用批处理方式处理所有邻居同时
            flat_len = edge_nei_len.reshape(-1, self.in_channels)
            flat_angle = edge_nei_angle.reshape(-1, self.in_channels)
            
            len_feat = self.lin_nei_len(flat_len).reshape(batch_size, 3, self.out_channels)
            angle_feat = self.lin_angle(flat_angle).reshape(batch_size, 3, self.out_channels)
            
            # 减少维度操作次数
            len_feat = len_feat.mean(dim=1)
            angle_feat = angle_feat.mean(dim=1)
            
            # 合并特征
            update = self.edge_update(torch.cat([edge_feat, len_feat, angle_feat], dim=-1))
        else:
            # 处理没有邻居信息的情况
            update = edge_feat
        
        # 使用门控机制决定更新多少信息
        gate_value = self.gate(torch.cat([edge_feat, update], dim=-1))
        
        # 门控更新
        edge_out = edge_feat + gate_value * update
        
        # 归一化
        edge_out = self.norm(edge_out)
        
        return F.relu(edge_out)
# ...existing code...

class DistanceAttenuation(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    # def forward(self, distances):
    #     cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
    #     cutoffs = cutoffs * (distances < self.cutoff_upper)
    #     return cutoffs

    # def forward(self, distances):
    #     # 使用 cos^2(x/2) 等价替换 0.5 * (cos(x) + 1)
    #     cutoffs = torch.cos(distances * math.pi / (2 * self.cutoff_upper)).pow(2)
    #     cutoffs = cutoffs * (distances < self.cutoff_upper)
    #     return cutoffs
    
    def forward(self, distances):
        # 使用更高效的实现
        cutoffs = torch.cos(distances * math.pi / (2 * self.cutoff_upper)).pow(2)
        # 用 where 代替乘法+条件判断，提高并行度
        cutoffs = torch.where(distances < self.cutoff_upper, cutoffs, torch.zeros_like(cutoffs))
        return cutoffs