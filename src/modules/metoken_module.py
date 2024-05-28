import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean
# from torch_geometric.nn.pool import knn_graph

import numpy as np
from .geometry import dihedrals, angles
from .affine_utils import Rotation, Rigid, quat_to_rot, rot_to_quat


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


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab=20):
        super().__init__()
        self.readout = nn.Linear(hidden_dim, vocab)
    
    def forward(self, h_V, batch_id=None):
        logits = self.readout(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs,logits
    

class StructureDecoder(nn.Module):
    def __init__(self, hidden_dim, num_decoder_layers=3, dropout=0.1):
        super().__init__()
        self.decoder = nn.ModuleList()
        for i in range(num_decoder_layers):
            self.decoder.append(
                GeneralE3GNN(1, hidden_dim, hidden_dim*2, dropout=dropout)
            )

    def forward(self, X, T, node_emb, edge_emb, edge_idx, batch_id):
        for layer in self.decoder:
            X, node_emb, edge_emb = layer(X.detach(), T, node_emb, edge_emb, edge_idx, batch_id)
        return X


class GeneralE3GNN(nn.Module):
    def __init__(self, gnn_layers, num_hidden, num_in, dropout=0.1, num_heads=None):
        super(GeneralE3GNN, self).__init__()
        self.W_logit = nn.Linear(num_hidden, 2)
        self.W_q = nn.Linear(num_hidden, 9)
        self.W_t = nn.Linear(num_hidden, 3)
        self.W_s_edge = nn.Linear(num_hidden, 2)
        self.W_atom = nn.Linear(num_hidden, 12)
        self.V_feat = nn.Linear(30, num_hidden)
        self.E_feat = nn.Linear(196, num_hidden)
        self.V_norm = nn.BatchNorm1d(num_hidden)
    
    def forward(self, X, T, node_emb, edge_emb, edge_idx, batch_id):
        row, col = edge_idx
        T_ji = T[col, None].invert().compose(T[row, None])
        R_s, t_s = self.message_passing_R_mat(T, T_ji, edge_emb, edge_idx, batch_id)
        T = Rigid(Rotation(R_s), t_s)
        self.local_atoms = self.W_atom(node_emb).view(-1, 4, 3)
        X = T[:,None].apply(self.local_atoms)
        return X, node_emb, edge_emb

    def message_passing_R_mat(self, T, T_ts, h_E, edge_idx, batch_id):
        quat_ts_init = T_ts._rots._rot_mats[:, 0].reshape(-1,9)
        t_ts_init = T_ts._trans[:, 0]
        all_num_node = batch_id.shape[0]
        s_edge = torch.sigmoid(self.W_s_edge(h_E)[..., None]).unbind(-2)
        d_scale = 10
        q_ts = s_edge[0] * quat_ts_init + (1.0 - s_edge[0]) * self.W_q(h_E)
        t_ts = s_edge[1] * t_ts_init + (1.0 - s_edge[1]) * d_scale * self.W_t(h_E)
        R_ts = self.avg_rotation(q_ts.reshape(-1,3,3))

        logit_ts = self.W_logit(h_E)
        R_s = T._rots._rot_mats
        t_s = T._trans
        R_s, t_s = self.equilibrate_transforms(R_s, t_s, R_ts, t_ts, logit_ts, edge_idx, all_num_node)
        return R_s, t_s

    def equilibrate_transforms(self, R_i_init, t_i_init, R_ji, t_ji, logit_ji, edge_idx, all_num_node):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        R_i_pred, t_i_pred = self.compose_transforms(R_i_init[dst_idx], t_i_init[dst_idx], R_ji, t_ji)
        probs = scatter_softmax(logit_ji, src_idx, dim=0)
        t_probs, R_probs = probs.unbind(-1)
        L = all_num_node
        t_avg = scatter_sum(t_i_pred*t_probs[:,None], src_idx, dim=0, dim_size=L)
        R_avg_unc = scatter_sum(R_i_pred*R_probs[:,None,None], src_idx, dim=0, dim_size=L)
        R_avg = self.avg_rotation(R_avg_unc)
        return R_avg, t_avg

    def avg_rotation(self, R_avg_unc, dither_eps = 1e-4):
        R_avg_unc2 = R_avg_unc + dither_eps * torch.randn_like(R_avg_unc)
        U, S, Vh = torch.linalg.svd(R_avg_unc2.float(), full_matrices=True)
        idx = 0
        while not self.svd_is_stable(S):
            R_avg_unc2 = R_avg_unc + dither_eps * torch.randn_like(R_avg_unc)
            U, S, Vh = torch.linalg.svd(R_avg_unc2.float(), full_matrices=True)
            idx += 1
            if idx>10:
                print('SVD is consistently unstable')
                break
        R_avg = U @ Vh
        d = torch.linalg.det(R_avg)
        d_expand = F.pad(d[..., None, None], (2, 0), value=1.0)
        Vh = Vh * d_expand
        R_avg = U @ Vh
        return R_avg

    def svd_is_stable(self, S):
        un_stable = (S.min()<1e-6) or (S ** 2).diff(dim=-1).abs().min()<1e-6
        return not un_stable

    def compose_transforms(self, R_a, t_a, R_b, t_b):
        R_composed = R_a @ R_b
        t_composed = t_a + (R_a @ t_b.unsqueeze(-1)).squeeze(-1)
        return R_composed, t_composed
    

class MCAttEGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_channel, in_edge_nf=0,
                 act_fn=nn.SiLU(), n_layers=4, residual=True, dropout=0.1, dense=False):
        super().__init__()
        '''
        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param n_channel: Number of channels of coordinates
        :param in_edge_nf: Number of features for the edge features
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param dropout: probability of dropout
        :param dense: if dense, then context states will be concatenated for all layers,
                      coordination will be averaged
        '''
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.dropout = nn.Dropout(dropout)

        self.linear_in = nn.Linear(in_node_nf, self.hidden_nf)

        self.dense = dense
        if dense:
            self.linear_out = nn.Linear(self.hidden_nf * (n_layers + 1), out_node_nf)
        else:
            self.linear_out = nn.Linear(self.hidden_nf, out_node_nf)

        for i in range(0, n_layers):
            self.add_module(f'gcl_{i}', MC_E_GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel,
                edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual, dropout=dropout
            ))
        self.out_layer = MC_E_GCL(
            self.hidden_nf, self.hidden_nf, self.hidden_nf, n_channel,
            edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual
        )
    
    def forward(self, h, x, ctx_edges, ctx_edge_attr=None):
        h = self.linear_in(h)
        h = self.dropout(h)

        ctx_states, ctx_coords, atts = [], [], []
        for i in range(0, self.n_layers):
            h, x = self._modules[f'gcl_{i}'](h, ctx_edges, x, edge_attr=ctx_edge_attr)
            ctx_states.append(h)
            ctx_coords.append(x)

        h, x = self.out_layer(h, ctx_edges, x, edge_attr=ctx_edge_attr)
        ctx_states.append(h)
        ctx_coords.append(x)
        if self.dense:
            h = torch.cat(ctx_states, dim=-1)
            x = torch.mean(torch.stack(ctx_coords), dim=0)
        h = self.dropout(h)
        h = self.linear_out(h)
        return h, x
    

class MC_E_GCL(nn.Module):
    """
    Multi-Channel E(n) Equivariant Convolutional Layer
    """

    def __init__(self, input_nf, output_nf, hidden_nf, n_channel, edges_in_d=0, act_fn=nn.SiLU(),
                 residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False,
                 dropout=0.1):
        super(MC_E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        self.dropout = nn.Dropout(dropout)

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + n_channel**2 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, n_channel, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        '''
        :param source: [n_edge, input_size]
        :param target: [n_edge, input_size]
        :param radial: [n_edge, n_channel, n_channel]
        :param edge_attr: [n_edge, edge_dim]
        '''
        radial = radial.reshape(radial.shape[0], -1)  # [n_edge, n_channel ^ 2]

        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        out = self.dropout(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        '''
        :param x: [bs * n_node, input_size]
        :param edge_index: list of [n_edge], [n_edge]
        :param edge_attr: [n_edge, hidden_size], refers to message from i to j
        :param node_attr: [bs * n_node, node_dim]
        '''
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))  # [bs * n_node, hidden_size]
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)  # [bs * n_node, input_size + hidden_size]
        out = self.node_mlp(agg)  # [bs * n_node, output_size]
        out = self.dropout(out)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        '''
        coord: [bs * n_node, n_channel, d]
        edge_index: list of [n_edge], [n_edge]
        coord_diff: [n_edge, n_channel, d]
        edge_feat: [n_edge, hidden_size]
        '''
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat).unsqueeze(-1)  # [n_edge, n_channel, d]

        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))  # [bs * n_node, n_channel, d]
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        '''
        h: [bs * n_node, hidden_size]
        edge_index: list of [n_row] and [n_col] where n_row == n_col (with no cutoff, n_row == bs * n_node * (n_node - 1))
        coord: [bs * n_node, n_channel, d]
        '''
        row, col = edge_index
        radial, coord_diff = coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)  # [n_edge, hidden_size]
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)    # [bs * n_node, n_channel, d]
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord
    

def coord2radial(edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]  # [n_edge, n_channel, d]
    radial = torch.bmm(coord_diff, coord_diff.transpose(-1, -2))  # [n_edge, n_channel, n_channel]
    # normalize radial
    radial = F.normalize(radial, dim=0)  # [n_edge, n_channel, n_channel]
    return radial, coord_diff

def unsorted_segment_sum(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    '''
    :param data: [n_edge, *dimensions]
    :param segment_ids: [n_edge]
    :param num_segments: [bs * n_node]
    '''
    expand_dims = tuple(data.shape[1:])
    result_shape = (num_segments, ) + expand_dims
    for _ in expand_dims:
        segment_ids = segment_ids.unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, *expand_dims)
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
