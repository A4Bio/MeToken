import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmModel
from src.tools import gather_nodes, _dihedrals, _get_rbf, _normalize, _quaternions, _orientations_coarse_gl_tuple
from src.modules.metoken_module import StructureEncoder, MeTokenDecoder, MLPDecoder

def _get_v_direct(X, eps=1e-6):
    B, N = X.shape[:2]
    V = X.clone()
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) 
    dX = X[:,1:,:] - X[:,:-1,:] # CA-N, C-CA, N-C, CA-N...
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)
    
    n_0, b_1 = n_0[:,::3,:], b_1[:,::3,:]
    X = X[:,::3,:]
    Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    Q = Q.view(list(Q.shape[:2]) + [9])
    Q = F.pad(Q, (0,0,0,1), 'constant', 0)
    Q = Q.view(list(Q.shape[:2]) + [3,3]).unsqueeze(2)
    dX_inner = V[:,:,[0,2,3],:] - X.unsqueeze(-2)
    dU_inner = torch.matmul(Q, dX_inner.unsqueeze(-1)).squeeze(-1)
    dU_inner = _normalize(dU_inner, dim=-1)
    V_direct = dU_inner.reshape(B,N,-1)
    return V_direct

def _get_e_direct_angle(X, E_idx, E_idx_select, eps=1e-6):
    V = X.clone()
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3) 
    dX = X[:,1:,:] - X[:,:-1,:]
    U = _normalize(dX, dim=-1)
    u_0, u_1 = U[:,:-2,:], U[:,1:-1,:]
    n_0 = _normalize(torch.cross(u_0, u_1), dim=-1)
    b_1 = _normalize(u_0 - u_1, dim=-1)
    
    n_0, b_1 = n_0[:,::3,:], b_1[:,::3,:]
    X = X[:,::3,:]
    Q = torch.stack((b_1, n_0, torch.cross(b_1, n_0)), 2)
    Q = Q.view(list(Q.shape[:2]) + [9])
    Q = F.pad(Q, (0,0,0,1), 'constant', 0)

    Q_neighbors = gather_nodes(Q, E_idx)
    X_neighbors = gather_nodes(V[:,:,1,:], E_idx)
    N_neighbors = gather_nodes(V[:,:,0,:], E_idx)
    C_neighbors = gather_nodes(V[:,:,2,:], E_idx)
    O_neighbors = gather_nodes(V[:,:,3,:], E_idx)

    Q = Q.view(list(Q.shape[:2]) + [3,3]).unsqueeze(2)
    Q_neighbors = Q_neighbors.view(list(Q_neighbors.shape[:3]) + [3,3])

    dX = torch.stack([X_neighbors,N_neighbors,C_neighbors,O_neighbors], dim=3) - X[:,:,None,None,:] 
    dU = torch.matmul(Q[:,:,:,None,:,:], dX[...,None]).squeeze(-1)
    B, N, K = dU.shape[:3]
    E_direct = _normalize(dU, dim=-1)
    E_direct = E_direct.reshape(B, N, K,-1)
    R = torch.matmul(Q.transpose(-1,-2), Q_neighbors)
    q = _quaternions(R)
    return E_idx_select(E_direct), E_idx_select(q)


class MeToken(nn.Module):
    def __init__(self, embedding_dim, num_ptm_types=26, num_per_type=128, commitment_cost=0.25, temperature=0.07):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_ptm_types = num_ptm_types
        self.num_per_type = num_per_type
        self.num_embeddings = num_ptm_types * num_per_type
        self.commitment_cost = commitment_cost
        self.temperature = temperature
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        # Initialize weights and normalize them
        initrange = 1.0 / self.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.embeddings.weight.data = F.normalize(self.embeddings.weight.data, p=2, dim=1)
        
    def forward(self, x, Q):  # Include Q in the forward pass  
        x = F.normalize(x, p=2, dim=-1)
        if Q is not None:
            encoding_indices = self.get_code_indices(x, Q)
            quantized = self.quantize(encoding_indices)

            q_latent_loss = F.mse_loss(quantized, x.detach())
            e_latent_loss = F.mse_loss(x, quantized.detach())
            uniform_loss = self.uniform_loss()
            loss = q_latent_loss + self.commitment_cost * e_latent_loss
            quantized = x + (quantized - x).detach().contiguous()
            return quantized, loss, uniform_loss, encoding_indices
        else:
            encoding_indices = self.get_code_indices_wo_ptm(x)
            quantized = self.quantize(encoding_indices)
            quantized = x + (quantized - x).detach().contiguous()
            return quantized, encoding_indices
    
    def get_code_indices(self, x, Q):
        batch_size, _ = x.shape
        encoding_indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Calculate distances only within the specified range for each PTM type
        for ptm_type in range(self.num_ptm_types):  # Assuming PTM types are 0 through 25
            ptm_mask = (Q.squeeze() == ptm_type)
            if ptm_mask.sum() == 0:
                continue  # Skip if no embeddings of this PTM type

            start_idx = ptm_type * self.num_per_type
            end_idx = (ptm_type + 1) * self.num_per_type
            ptm_embeddings = self.embeddings.weight[start_idx:end_idx]

            # Calculate distances only for the current PTM type embeddings
            distances = (
                torch.sum(x[ptm_mask] ** 2, dim=1, keepdim=True) +
                torch.sum(ptm_embeddings ** 2, dim=1) -
                2. * torch.matmul(x[ptm_mask], ptm_embeddings.t())
            )

            # Find the nearest embeddings within this segment
            ptm_encoding_indices = torch.argmin(distances, dim=1)
            encoding_indices[ptm_mask] = ptm_encoding_indices + start_idx

        return encoding_indices
    
    def get_code_indices_wo_ptm(self, x):
        distances = (
            torch.sum(x ** 2, dim=-1, keepdim=True) +
            torch.sum(F.normalize(self.embeddings.weight, p=2, dim=-1) ** 2, dim=1) -
            2. * torch.matmul(x, F.normalize(self.embeddings.weight.t(), p=2, dim=0))
        )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices
    
    def quantize(self, encoding_indices):
        return F.normalize(self.embeddings(encoding_indices), p=2, dim=-1)

    def uniform_loss(self):
        # Normalize embeddings for cosine similarity calculation
        normalized_embeddings = F.normalize(self.embeddings.weight, p=2, dim=1)

        # Generating indices for 10% sampling from each PTM type segment
        sampled_num = int(0.1 * self.num_per_type)
        all_indices = torch.arange(self.num_embeddings, device=self.embeddings.weight.device)
        reshaped_indices = all_indices.view(self.num_ptm_types, self.num_per_type)
        sampled_indices = reshaped_indices[:, torch.randperm(self.num_per_type)[:sampled_num]].view(-1)
        sampled_embeddings = normalized_embeddings[sampled_indices]

        # Calculate cosine similarity matrix for the sampled embeddings
        sim_matrix = torch.mm(sampled_embeddings, sampled_embeddings.t())

        # Create labels for contrastive loss based on sampled index
        labels = (sampled_indices / self.num_per_type).long()  # PTM type index based on position in codebook
        labels = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Calculate contrastive loss using label information
        mask = torch.eye(len(sampled_indices), device=sampled_indices.device).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))  # Remove self-similarity
        sim_exp = torch.exp(sim_matrix / self.temperature)
        sum_exp = torch.sum(sim_exp, dim=1, keepdim=True)

        pos_exp = sim_exp * labels.float()
        pos_sum = torch.sum(pos_exp, dim=1, keepdim=True)

        contrastive_loss = -torch.log(pos_sum / sum_exp).mean()
        return contrastive_loss


class MeTokenPro_Model(nn.Module):
    def __init__(self, args):
        """ Graph labeling network """
        super(MeTokenPro_Model, self).__init__()
        self.args = args
        # {0: '<cls>', 1: '<pad>', 2: '<eos>', 3: '<unk>', 4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C', 24: 'X', 25: 'B', 26: 'U', 27: 'Z', 28: 'O', 29: '.', 30: '-', 31: '<null_1>', 32: '<mask>'}
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/tancheng/model_zoom/transformers")
        # self.esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/tancheng/model_zoom/transformers")
        # for param in self.esm.parameters():
        #     param.requires_grad = False
        self.wo_esm = nn.Embedding(len(self.tokenizer._token_to_id), self.args.hidden_dim)

        # self.W_s = nn.Sequential(
        #     nn.Linear(self.esm.pooler.dense.in_features, args.hidden_dim),
        #     nn.SiLU(),
        #     nn.BatchNorm1d(args.hidden_dim),
        #     nn.Linear(args.hidden_dim, args.hidden_dim)
        # )

        self.node_embed = nn.Linear(117, args.node_features)
        self.node_norm = nn.BatchNorm1d(args.node_features)
        self.W_v = nn.Sequential(
            nn.Linear(args.node_features, args.hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )

        self.W_node_gate = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )
        self.W_node_emb = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )

        self.seq_k_emb = nn.Linear(272, args.edge_features)
        self.seq_k_norm = nn.BatchNorm1d(args.edge_features)
        self.W_seq_k = nn.Sequential(
            nn.Linear(args.edge_features, args.hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )

        self.str_k_emb = nn.Linear(272, args.edge_features)
        self.str_k_norm = nn.BatchNorm1d(args.edge_features)
        self.W_str_k = nn.Sequential(
            nn.Linear(args.edge_features, args.hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )

        self.str_r_emb = nn.Linear(272, args.edge_features)
        self.str_r_norm = nn.BatchNorm1d(args.edge_features)
        self.W_str_r = nn.Sequential(
            nn.Linear(args.edge_features, args.hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(args.hidden_dim),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )

        self.encoder = StructureEncoder(args.hidden_dim, args.num_encoder_layers, args.dropout, args.module_type)
        self.metoken = MeToken(args.hidden_dim)
        self.decoder = MeTokenDecoder(args.hidden_dim, args.num_encoder_layers, args.dropout, args.module_type)

        if self.args.module_type == 0:
            self.predictor = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.vocab)
            )
        elif self.args.module_type in [51,52,53,71]:
            self.predictor1 = StructureEncoder(args.hidden_dim, num_encoder_layers=1, dropout=args.dropout, module_type=args.module_type)
            self.predictor2 = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.vocab)
            )
            self.predictor3 = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.hidden_dim)
            )
            self.predictor4 = nn.Sequential(
                nn.Linear(args.hidden_dim*2, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.hidden_dim)
            )
        elif self.args.module_type in [54,55,56,74]:
            self.predictor1 = StructureEncoder(args.hidden_dim, num_encoder_layers=3, dropout=args.dropout, module_type=args.module_type)
            self.predictor2 = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.vocab)
            )
            self.predictor3 = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.hidden_dim)
            )
            self.predictor4 = nn.Sequential(
                nn.Linear(args.hidden_dim*2, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.hidden_dim)
            )
        elif self.args.module_type in [57,58,59,77]:
            self.predictor1 = StructureEncoder(args.hidden_dim, num_encoder_layers=6, dropout=args.dropout, module_type=args.module_type)
            self.predictor2 = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.vocab)
            )
            self.predictor3 = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.hidden_dim)
            )
            self.predictor4 = nn.Sequential(
                nn.Linear(args.hidden_dim*2, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.hidden_dim)
            )
        elif self.args.module_type in [60,61,62,80]:
            self.predictor1 = StructureEncoder(args.hidden_dim, num_encoder_layers=9, dropout=args.dropout, module_type=args.module_type)
            self.predictor2 = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.vocab)
            )
            self.predictor3 = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.hidden_dim)
            )
            self.predictor4 = nn.Sequential(
                nn.Linear(args.hidden_dim*2, args.hidden_dim*2),
                nn.SiLU(),
                nn.Linear(args.hidden_dim*2, args.hidden_dim)
            )
        


        self.token_idx_emb = nn.Embedding(26*128, args.hidden_dim)

        if args.pretrain == 0:
            for module in [self.encoder, self.metoken]:
                for param in module.parameters():
                    param.requires_grad = False            

    def get_seq_knearest(self, B, L, mask, k=11):
        seq_k_idx = []
        for i in range(L):
            row = [(i + j - k) % L for j in range(2 * k + 1)]
            seq_k_idx.append(row)
        return torch.tensor(seq_k_idx, device=mask.device).repeat(B, 1, 1)

    def get_str_knearest(self, x, mask, k=30, eps=1e-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(x,1) - torch.unsqueeze(x,2)
        D = (1. - mask_2D)*1e6 + mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        _, topk_idx = torch.topk(D_adjust, min(k, D_adjust.shape[-1]), dim=-1, largest=False)
        return topk_idx
    
    def get_r_ball_neighbors(self, x, mask, radius=10.0, max_k=100, eps=1e-6):
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(x, 1) - torch.unsqueeze(x, 2)
        D = (1. - mask_2D)*1e6 + mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        within_r = D_adjust < radius
        within_r[:, :, 0] = True
        within_r = (mask_2D.long() & within_r).bool()
        (b, i, j), _= torch.nonzero(within_r, as_tuple=False).max(0)
        L = min(max(i, j) + 1, max_k) # max neighbors = 100
        r_eidx = torch.arange(L).unsqueeze(0).repeat(x.shape[0], x.shape[1], 1)
        return within_r[:, :, :L], r_eidx.to(x.device)
    
    def idx_transform(self, mask, eidx, eidx_mask):
        B, N = mask.shape
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B,1,1) + eidx
        src = torch.masked_select(src, eidx_mask).view(1,-1)
        dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(eidx_mask)
        dst = torch.masked_select(dst, eidx_mask).view(1,-1)
        eidx = torch.cat((dst, src), dim=0).long()
        return eidx
            
    def forward(self, batch, pretrain=False):
        S, X, Q, mask = batch['S'], batch['X'], batch['Q'], batch['mask']
        atom_N, atom_Ca, atom_C, atom_O = X.unbind(2)

        # build micro environment
        # - seq k-nearest
        seq_k_eidx = self.get_seq_knearest(S.shape[0], S.shape[-1], mask)
        # - str k-nearest
        str_k_eidx = self.get_str_knearest(atom_Ca, mask)
        # - str r-ball
        str_r_eidx_mask, str_r_eidx = self.get_r_ball_neighbors(atom_Ca, mask)

        node_mask_select = lambda x: torch.masked_select(x, (mask == 1).unsqueeze(-1)).reshape(-1, x.shape[-1])
        seq_k_eidx_mask = (mask.unsqueeze(-1) * gather_nodes(mask.unsqueeze(-1), seq_k_eidx).squeeze(-1)) == 1
        str_k_eidx_mask = (mask.unsqueeze(-1) * gather_nodes(mask.unsqueeze(-1), str_k_eidx).squeeze(-1)) == 1

        seq_k_edix_select = lambda x: torch.masked_select(x, seq_k_eidx_mask.unsqueeze(-1)).reshape(-1, x.shape[-1])
        str_k_edix_select = lambda x: torch.masked_select(x, str_k_eidx_mask.unsqueeze(-1)).reshape(-1, x.shape[-1])
        str_r_edix_select = lambda x: torch.masked_select(x, str_r_eidx_mask.unsqueeze(-1)).reshape(-1, x.shape[-1])

        # node sequence emb
        # node_seq_emb = self.W_s(node_mask_select(self.esm(input_ids=S).last_hidden_state))
        node_seq_emb = node_mask_select(self.wo_esm(S))
        # node structure emb
        # - node angle, BxNx12
        v_angles = node_mask_select(_dihedrals(X))
        # - node dist,  BxNx(6x16)
        v_dists = []
        for pair in ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']:
            atom_a, atom_b = pair.split('-')
            v_dists.append(node_mask_select(_get_rbf(vars()['atom_' + atom_a], vars()['atom_' + atom_b]).squeeze()))
        v_dists = torch.cat(v_dists, dim=-1)
        # - node direct, BxNx9
        v_directs = node_mask_select(_get_v_direct(X))
        # node emb summary
        node_str_emb = self.W_v(self.node_norm(self.node_embed(torch.cat([v_angles, v_dists, v_directs], dim=-1))))
        node_gate = self.W_node_gate(torch.cat([node_seq_emb, node_str_emb], dim=-1))
        node_emb = self.W_node_emb(torch.cat([node_seq_emb, node_str_emb], dim=-1))
        node_emb = torch.sigmoid(node_gate) * node_emb

        # - edge dist, BxNxKx256
        seq_k_dists, str_k_dists, str_r_dists = [], [], []
        for pair in ['Ca-Ca', 'Ca-C', 'C-Ca', 'Ca-N', 'N-Ca', 'Ca-O', 'O-Ca', 'C-C', 'C-N', 'N-C', 'C-O', 'O-C', 'N-N', 'N-O', 'O-N', 'O-O']:
            atom_a, atom_b = pair.split('-')
            seq_k_dists.append(seq_k_edix_select(_get_rbf(vars()['atom_' + atom_a], vars()['atom_' + atom_b], seq_k_eidx)))
            str_k_dists.append(str_k_edix_select(_get_rbf(vars()['atom_' + atom_a], vars()['atom_' + atom_b], str_k_eidx)))
            str_r_dists.append(str_r_edix_select(_get_rbf(vars()['atom_' + atom_a], vars()['atom_' + atom_b], str_r_eidx)))
        seq_k_dists = torch.cat(seq_k_dists, dim=-1)
        str_k_dists = torch.cat(str_k_dists, dim=-1)
        str_r_dists = torch.cat(str_r_dists, dim=-1)
        # - edge direct BxNxKx12 and edge angle, BxNxKx4
        seq_k_directs, seq_k_angles = _get_e_direct_angle(X, seq_k_eidx, seq_k_edix_select)
        str_k_directs, str_k_angles = _get_e_direct_angle(X, str_k_eidx, str_k_edix_select)
        str_r_directs, str_r_angles = _get_e_direct_angle(X, str_r_eidx, str_r_edix_select)
        # edge emb summary
        seq_k_emb = self.W_seq_k(self.seq_k_norm(self.seq_k_emb(torch.cat([seq_k_dists, seq_k_directs, seq_k_angles], dim=-1))))
        str_k_emb = self.W_str_k(self.str_k_norm(self.str_k_emb(torch.cat([str_k_dists, str_k_directs, str_k_angles], dim=-1))))
        str_r_emb = self.W_str_r(self.str_r_norm(self.str_r_emb(torch.cat([str_r_dists, str_r_directs, str_r_angles], dim=-1))))

        seq_k_eidx = self.idx_transform(mask, seq_k_eidx, seq_k_eidx_mask)
        str_k_eidx = self.idx_transform(mask, str_k_eidx, str_k_eidx_mask)
        str_r_eidx = self.idx_transform(mask, str_r_eidx, str_r_eidx_mask)

        sparse_idx = mask.nonzero()  # index of non-zero values
        batch_id = sparse_idx[:,0]

        if pretrain == True:
            node_emb = self.encoder(node_emb, seq_k_emb, seq_k_eidx, str_k_emb, str_k_eidx, str_r_emb, str_r_eidx, batch_id)

            token_emb, e_q_loss, u_loss, encoding_indices = self.metoken(node_emb, torch.masked_select(Q, mask == 1))
            node_emb_recon = self.decoder(token_emb, seq_k_eidx, str_k_eidx, str_r_eidx, batch_id)
            recon_loss = F.mse_loss(node_emb, node_emb_recon)

            loss = e_q_loss + recon_loss + 0.1 * u_loss
            return loss
        else:
            if self.args.module_type == 0:
                node_emb = self.encoder(node_emb, seq_k_emb, seq_k_eidx, str_k_emb, str_k_eidx, str_r_emb, str_r_eidx, batch_id)
                token_emb, encoding_indices = self.metoken(node_emb, None)
                log_probs = F.log_softmax(self.predictor(token_emb + self.token_idx_emb(encoding_indices)), dim=-1)

                batch.update({'Q': torch.masked_select(Q, mask == 1), 'mask': torch.masked_select(mask, mask == 1)})
                return {'log_probs': log_probs}
            elif self.args.module_type in [51, 54, 57, 60]:
                node_emb = self.encoder(node_emb, seq_k_emb, seq_k_eidx, str_k_emb, str_k_eidx, str_r_emb, str_r_eidx, batch_id)
                token_emb, encoding_indices = self.metoken(node_emb, None)

                pred_emb = self.predictor1(node_emb, seq_k_emb, seq_k_eidx, str_k_emb, str_k_eidx, str_r_emb, str_r_eidx, batch_id)
                node_emb = torch.sigmoid(token_emb) * pred_emb
                log_probs = F.log_softmax(self.predictor2(node_emb), dim=-1)

                batch.update({'Q': torch.masked_select(Q, mask == 1), 'mask': torch.masked_select(mask, mask == 1)})
                return {'log_probs': log_probs}
            elif self.args.module_type in [71, 74, 77, 80]:
                node_emb = self.encoder(node_emb, seq_k_emb, seq_k_eidx, str_k_emb, str_k_eidx, str_r_emb, str_r_eidx, batch_id)
                token_emb, encoding_indices = self.metoken(node_emb, None)

                pred_emb = self.predictor1(node_emb, seq_k_emb, seq_k_eidx, str_k_emb, str_k_eidx, str_r_emb, str_r_eidx, batch_id)
                node_emb = pred_emb
                log_probs = F.log_softmax(self.predictor2(node_emb), dim=-1)

                batch.update({'Q': torch.masked_select(Q, mask == 1), 'mask': torch.masked_select(mask, mask == 1)})
                return {'log_probs': log_probs}