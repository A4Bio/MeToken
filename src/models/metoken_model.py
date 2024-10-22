import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.tools import gather_nodes, _dihedrals, _get_rbf, _normalize, _quaternions
from src.modules.metoken_module import StructureEncoder, MeTokenDecoder


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
        self.num_embeddings = num_ptm_types * num_per_type
        self.num_per_type = num_per_type

        self.num_per_type_lst = num_ptm_types * [num_per_type] 
        
        self.commitment_cost = commitment_cost
        self.temperature = temperature
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        with torch.no_grad():
            self.embeddings.weight.data.normal_(0, 1)
            self.embeddings.weight.data = F.normalize(self.embeddings.weight.data, p=2, dim=1)

    def _get_sub_codebook(self, index):
        sub_codebook_sizes = torch.tensor(self.num_per_type_lst, device=index.device)
        cumulative_sizes = torch.cumsum(sub_codebook_sizes, dim=0)
        return torch.searchsorted(cumulative_sizes, index, right=True)
    
    def quantize(self, x, codebook, tau):
        dot_product = torch.matmul(x, codebook.t()) / tau
        similarity = F.softmax(dot_product, dim=-1)
        z_q = torch.matmul(similarity, codebook)
        encoding_indices = torch.argmax(similarity, dim=-1)
        return z_q, encoding_indices
        
    def forward(self, x, Q, tau=1.):
        if Q is not None:
            quantized, encoding_indices = self.get_code_indices(x, Q, tau)
            loss = 0.
            uniform_loss = self.uniform_loss()
            return quantized, loss, uniform_loss, encoding_indices
        else:
            quantized, encoding_indices = self.get_code_indices_wo_ptm(x, tau)
            return quantized, encoding_indices
    
    def get_code_indices(self, x, Q, tau):
        batch_size, _ = x.shape
        encoding_indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        quantized = torch.zeros_like(x)
        
        start_idx = 0
        for ptm_type in range(self.num_ptm_types):
            end_idx = start_idx + self.num_per_type_lst[ptm_type]
            ptm_mask = (Q.squeeze() == ptm_type)
            if ptm_mask.any():
                ptm_quantized, ptm_encoding_indices = self.quantize(x[ptm_mask], self.embeddings.weight[start_idx:end_idx], tau)
                quantized[ptm_mask] = ptm_quantized
                encoding_indices[ptm_mask] = ptm_encoding_indices + start_idx
            start_idx = end_idx
        return quantized, encoding_indices
    
    def get_code_indices_wo_ptm(self, x, tau):
        quantized, encoding_indices = self.quantize(x, self.embeddings.weight, tau)
        return quantized, encoding_indices
    
    def uniform_loss(self):
        normalized_embeddings = F.normalize(self.embeddings.weight, p=2, dim=1)
        sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())

        indices = torch.arange(self.num_embeddings, device=self.embeddings.weight.device)
        labels = self._get_sub_codebook(indices)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)

        mask = torch.eye(len(indices), device=indices.device).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))
        sim_exp = torch.exp(sim_matrix / self.temperature)
        sum_exp = torch.sum(sim_exp, dim=1, keepdim=True)
        pos_sim_exp = sim_exp * labels_eq.float()
        pos_sum_exp = torch.sum(pos_sim_exp, dim=1, keepdim=True)
        return -torch.log(pos_sum_exp / sum_exp).mean()


class MeToken_Model(nn.Module):
    def __init__(self, args):
        super(MeToken_Model, self).__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.wo_esm = nn.Embedding(len(self.tokenizer._token_to_id), self.args.hidden_dim)
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
        self.masked_metoken = MeToken(args.hidden_dim)
        self.decoder = MeTokenDecoder(args.hidden_dim, args.num_encoder_layers, args.dropout, args.module_type)

        self.predictor1 = StructureEncoder(args.hidden_dim, num_encoder_layers=9, dropout=args.dropout, module_type=args.module_type)
        self.predictor2 = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim*2),
            nn.SiLU(),
            nn.Linear(args.hidden_dim*2, args.vocab)
        )

        self.token_idx_emb = nn.Embedding(26*128, args.hidden_dim)
        codebook = self.metoken.embeddings.weight
        codebook_mask = torch.ones(len(codebook), dtype=torch.int32,device='cuda')
        indices = torch.randperm(len(codebook))[:args.codebook_masknum]
        codebook_mask[indices] = 0
        self.codebook_mask = codebook_mask

        self.pretrain_mode = args.pretrain_mode
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
            
    def forward(self, batch, pretrain=False, mode='train', tau=1.0):
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
            if self.pretrain_mode==1:
                pass
            else:
                token_emb, e_q_loss, u_loss, encoding_indices = self.metoken(node_emb, torch.masked_select(Q, mask == 1), tau)
                pred_sub_codebook = self.metoken._get_sub_codebook(encoding_indices)
                node_emb_recon = self.decoder(token_emb, seq_k_eidx, str_k_eidx, str_r_eidx, batch_id)
                recon_loss = F.mse_loss(node_emb, node_emb_recon)
                vq_loss = e_q_loss + recon_loss + 0.1 * u_loss
                    
                codebook=self.metoken.embeddings.weight
                codebook_mask=self.codebook_mask
                masked_codebook=codebook.clone()
                expanded_mask = codebook_mask.unsqueeze(1).expand(-1, 128)
                masked_codebook = masked_codebook * expanded_mask
                with torch.no_grad():
                    self.masked_metoken.embeddings.weight.copy_(masked_codebook)
                    masked_tokenemb,masked_embindex=self.masked_metoken(node_emb,None,tau)
                    masked_recon = self.decoder(masked_tokenemb, seq_k_eidx, str_k_eidx, str_r_eidx, batch_id)

                mcm_loss=0
                gamma=1.0
                for i in range(0,len(node_emb)):
                    index=masked_embindex[i]
                    recon=masked_recon[i]
                    if codebook_mask[index]==0:
                        mcm_loss+=(F.cosine_embedding_loss(node_emb[i],recon,torch.ones(1,device='cuda'))**gamma)
                    else:
                        mcm_loss+=torch.tensor(0,device='cuda')
                tot_masked=len(codebook_mask==0)
                mcm_loss=mcm_loss/tot_masked

                eta=10
                loss=vq_loss+eta*mcm_loss
                if mode=="train":
                    return loss
                batch.update({'Q': torch.masked_select(Q, mask == 1), 'mask': torch.masked_select(mask, mask == 1)})
                return loss, pred_sub_codebook
                
        else:
            if self.args.module_type in [94]:# 94 as default
                node_emb = self.encoder(node_emb, seq_k_emb, seq_k_eidx, str_k_emb, str_k_eidx, str_r_emb, str_r_eidx, batch_id)
                token_emb, encoding_indices = self.metoken(node_emb, None, tau=1e-4)

                pred_emb = self.predictor1(token_emb+node_emb, seq_k_emb, seq_k_eidx, str_k_emb, str_k_eidx, str_r_emb, str_r_eidx, batch_id)
                log_probs = F.log_softmax(self.predictor2(pred_emb), dim=-1)

                batch.update({'Q': torch.masked_select(Q, mask == 1), 'mask': torch.masked_select(mask, mask == 1),'S': torch.masked_select(S, mask == 1)})
                return {'log_probs': log_probs,"codebook": self.metoken.embeddings.weight,"node_emb":node_emb,"token_emb":token_emb,"pred_emb":pred_emb,"token_index":encoding_indices}
            
