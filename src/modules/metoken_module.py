import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean
import numpy as np


class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=1, edge_drop=0.0, output_mlp=True):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        self.output_mlp = output_mlp
        
        self.W_V = nn.Sequential(nn.Linear(num_in, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden)
        )
        self.Bias = nn.Sequential(
            nn.Linear(num_hidden*3, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden,num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden,num_heads)
        )
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)

    def forward(self, h_V, h_E, center_id):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        
        w = self.Bias(torch.cat([h_V[center_id], h_E],dim=-1)).view(E, n_heads, 1) 
        attend_logits = w/np.sqrt(d) 

        V = self.W_V(h_E).view(-1, n_heads, d) 
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        h_V = scatter_sum(attend*V, center_id, dim=0).view([-1, self.num_hidden])

        if self.output_mlp:
            h_V_update = self.W_O(h_V)
        else:
            h_V_update = h_V
        return h_V_update


class GeneralGNN(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(GeneralGNN, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(3)])
        self.attention = NeighborAttention(num_hidden, num_in, num_heads=4) 
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.SiLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        dh = self.attention(h_V, torch.cat([h_E, h_V[dst_idx]], dim=-1), src_idx)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        return h_V


class StructureEncoder(nn.Module):
    def __init__(self,  hidden_dim, num_encoder_layers=3, dropout=0, module_type=0):
        """ Graph labeling network """
        super(StructureEncoder, self).__init__()
        encoder_layers = []        
        module = GeneralGNN
        self.module_type = module_type
        for _ in range(num_encoder_layers):
            encoder_layers.append(
                module(hidden_dim, hidden_dim*2, dropout=dropout),
            )
        self.encoder_layers = nn.Sequential(*encoder_layers)

    def forward(self, node_emb, seq_k_emb, seq_k_eidx, str_k_emb, str_k_eidx, str_r_emb, str_r_eidx, batch_id):
        for layer in self.encoder_layers:
            node_emb_seq_k = layer(node_emb, seq_k_emb, seq_k_eidx, batch_id)
            node_emb_str_k = layer(node_emb, str_k_emb, str_k_eidx, batch_id)
            node_emb_str_r = layer(node_emb, str_r_emb, str_r_eidx, batch_id)
            node_emb = node_emb_seq_k * torch.sigmoid(node_emb_seq_k) + node_emb_str_k * torch.sigmoid(node_emb_str_k) + node_emb_str_r * torch.sigmoid(node_emb_str_r)
        return node_emb


class MeTokenGNN(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super(MeTokenGNN, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(3)])
        self.W_V = nn.Sequential(nn.Linear(num_in, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.SiLU(),
            nn.Linear(num_hidden, num_hidden)
        )
        self.dense = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.SiLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )

    def forward(self, h_V, edge_idx, batch_id):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        dh = self.W_V(torch.cat([h_V[src_idx], h_V[dst_idx]], dim=-1))
        dh = scatter_mean(dh, src_idx, dim=0)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        return h_V
    

class MeTokenDecoder(nn.Module):
    def __init__(self,  hidden_dim, num_decoder_layers=3, dropout=0, module_type=0):
        """ Graph labeling network """
        super(MeTokenDecoder, self).__init__()
        encoder_layers = []        
        module = MeTokenGNN
        self.module_type = module_type
        for _ in range(num_decoder_layers):
            encoder_layers.append(
                module(hidden_dim, hidden_dim*2, dropout=dropout),
            )
        self.encoder_layers = nn.Sequential(*encoder_layers)

    def forward(self, node_emb, seq_k_eidx, str_k_eidx, str_r_eidx, batch_id):
        for layer in self.encoder_layers:
            node_emb_seq_k = layer(node_emb, seq_k_eidx, batch_id)
            node_emb_str_k = layer(node_emb, str_k_eidx, batch_id)
            node_emb_str_r = layer(node_emb, str_r_eidx, batch_id)
            node_emb = node_emb_seq_k * torch.sigmoid(node_emb_seq_k) + node_emb_str_k * torch.sigmoid(node_emb_str_k) + node_emb_str_r * torch.sigmoid(node_emb_str_r)
        return node_emb