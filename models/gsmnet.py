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

from models.grit.encoder.rrwp_encoder import RRWPLinearNodeEncoder, RRWPLinearEdgeEncoder, full_edge_index
from models.grit.layer.grit_layer import GritTransformerLayer, pyg_softmax, StateFormer


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

    mamba_d_state: int = 64
    mamba_d_conv: int = 2
    mamba_expand: int = 2
    mamba_dt_rank: str = "auto"  
    mamba_dt_min: float = 0.001
    mamba_dt_max: float = 0.1
    mamba_dropout: float = 0.1

    class Config:
        """Configure model settings behavior."""
        env_prefix = "jv_model"


class DDMPNN(MessagePassing):

    def __init__(self, fc_features, cutoff_upper=5.0):
        super(DDMPNN, self).__init__(node_dim=0)
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

        self.edge_update_layer = EdgeUpdateLayer(
            in_channels=config.fc_features,
            out_channels=config.fc_features,
            
        )

        self.local_modules = nn.ModuleList(
            [
                DDMPNN(config.fc_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.rrwp_rel_encoder = RRWPLinearEdgeEncoder(config.ksteps, config.fc_features,
                                                      pad_to_full_graph=True,
                                                      add_node_attr_as_self_loop=False,
                                                      fill_value=0.)

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

        edge_features = torch.cat([edge_features, edge_dir], dim=-1)  # [num_edges, fc_features+3]
        edge_features = self.edge_dir_mlp(edge_features)  

        edge_nei_len = None
        edge_nei_angle = None
        if hasattr(data, 'edge_nei') and data.edge_nei is not None:
            edge_nei_len = -0.75 / torch.norm(data.edge_nei, dim=-1)  # [num_edges, 3]
            edge_nei_angle = bond_cosine(data.edge_attr, data.edge_nei)  # [num_edges, 3]
            num_edge = edge_features.shape[0]
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
                    node_features, rrwp_edge_attr_enc = self.global_modules[i](node_features, rrwp_edge_index_enc,
                                                                                    rrwp_edge_attr_enc, node_nums,
                                                                                    log_deg, inf_edge_index,
                                                                        inf_edge_features)
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

def bond_cosine(r1, r2):
    r1_expand = r1.unsqueeze(1)  # [num_edges, 1, 3]
    dot_product = torch.einsum('eij,eij->ei', r1_expand, r2)
    norm_r1 = torch.norm(r1, dim=-1, keepdim=True)
    norm_r2 = torch.norm(r2, dim=-1)
    cos = dot_product / (norm_r1 * norm_r2 + 1e-8)
    return torch.clamp(cos, -1, 1)

class EdgeUpdateLayer(nn.Module):
    
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
        
        self.lin_edge = nn.Linear(in_channels, out_channels)
        
        self.lin_nei_len = nn.Linear(in_channels, out_channels)
        
        self.lin_angle = nn.Linear(in_channels, out_channels)
        
        self.edge_update = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        self.norm = nn.LayerNorm(out_channels)
        
        self.gate = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.Sigmoid()
        )
    
    def forward(self, edge_features, edge_nei_len, edge_nei_angle):
        """
        edge_features: [num_edges, hidden_dim]
        edge_nei_len: [num_edges, 3, hidden_dim] 
        edge_nei_angle: [num_edges, 3, hidden_dim] 
        """
        batch_size = edge_features.shape[0]
        edge_feat = self.lin_edge(edge_features)
        
        if edge_nei_len is not None and edge_nei_angle is not None:
            flat_len = edge_nei_len.reshape(-1, self.in_channels)
            flat_angle = edge_nei_angle.reshape(-1, self.in_channels)
            
            len_feat = self.lin_nei_len(flat_len).reshape(batch_size, 3, self.out_channels)
            angle_feat = self.lin_angle(flat_angle).reshape(batch_size, 3, self.out_channels)

            len_feat = len_feat.mean(dim=1)
            angle_feat = angle_feat.mean(dim=1)

            update = self.edge_update(torch.cat([edge_feat, len_feat, angle_feat], dim=-1))
        else:
            update = edge_feat

        gate_value = self.gate(torch.cat([edge_feat, update], dim=-1))
        
        edge_out = edge_feat + gate_value * update

        edge_out = self.norm(edge_out)
        
        return F.relu(edge_out)

class DistanceAttenuation(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super().__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
    
    def forward(self, distances):
        cutoffs = torch.cos(distances * math.pi / (2 * self.cutoff_upper)).pow(2)
        cutoffs = torch.where(distances < self.cutoff_upper, cutoffs, torch.zeros_like(cutoffs))
        return cutoffs