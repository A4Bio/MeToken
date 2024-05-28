import torch.nn as nn
from torch_geometric.data import  Data,Batch
import torch.nn.functional as F
from src.modules.dtl_dephossite_module import *

class DTL_DephosSite_Model(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.vocab=args.vocab
        self.embedding=nn.Embedding(num_embeddings=24,embedding_dim=21) # [16,500,21]
        self.bilstm=BiLSTM(embedding_dim=21,output_dim=32)
        self.mlp=MLP(in_dim=32,out_dim=self.vocab)

    def forward(self,batch):
        batch=batch["S"] # [16,500]
        batch_len = batch.shape[1]
        batch=F.pad(batch, (0, 500-batch_len), "constant", 0)
        batchsize=batch.shape[0]
        embedded=self.embedding(batch)
        lstmed=self.bilstm(embedded)
        output=self.mlp(lstmed)
        log_probs = F.log_softmax(output, dim=-1)[:, :batch_len]
        return {'log_probs': log_probs}
