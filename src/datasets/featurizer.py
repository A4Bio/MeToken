import torch
import numpy as np
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")


def featurize(batch):
    """ Pack and pad batch into torch tensors """
    # alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    batch = [one for one in batch if one is not None]
    B = len(batch)
    if B==0:
        return None
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max([len(b['seq']) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    Q = np.zeros([B, L_max], dtype=np.int32)
    score = np.ones([B, L_max]) * 100.0
    chain_mask = np.zeros([B, L_max])
    chain_encoding = np.zeros([B, L_max])
    
    for i, b in enumerate(batch):
        x = np.stack([b["coords_chain_A"][c] for c in ['N_chain_A', 'CA_chain_A', 'C_chain_A', 'O_chain_A']], 1)
        
        l = len(b['seq'])
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        indices = np.array(b['ptm'])
        Q[i, :l] = indices
        S[i, :l] = np.array(tokenizer.encode(b['seq'], add_special_tokens=False))

    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    numbers = np.sum(mask, axis=1).astype(np.int32)
    S_new = np.zeros_like(S)
    Q_new = np.zeros_like(Q)
    X_new = np.zeros_like(X)+np.nan
    for i, n in enumerate(numbers):
        X_new[i,:n,::] = X[i][mask[i]==1]
        S_new[i,:n] = S[i][mask[i]==1]
        Q_new[i,:n] = Q[i][mask[i]==1]

    X = X_new
    S = S_new
    Q = Q_new
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    S = torch.from_numpy(S).to(dtype=torch.long)
    score = torch.from_numpy(score).float()
    X = torch.from_numpy(X).to(dtype=torch.float32)
    Q = torch.from_numpy(Q).to(dtype=torch.long)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    lengths = torch.from_numpy(lengths)
    chain_mask = torch.from_numpy(chain_mask)
    chain_encoding = torch.from_numpy(chain_encoding)
    
    return {"id": [b['id'] for b in batch],
            "X": X,
            "Q": Q, # sequence of target (PTM here)
            "S": S, # sequence of protein AAs
            "score": score,
            "mask": mask,
            "lengths": lengths,
            "chain_mask": chain_mask,
            "chain_encoding": chain_encoding}