import torch.nn as nn
from src.modules.deepphos_module import *

class DeepPhos_model(nn.Module):
    def __init__(self,args, **kwargs):
        super().__init__()
        dropout=args.dropout
        vocab=args.vocab
        self.dccnn1=DC_CNN(dropout)
        self.dccnn2=DC_CNN(dropout)
        self.dccnn3=DC_CNN(dropout)  # [32,500,16]
        self.bcl=Intra_BCL(out_channel=500,p_dropout=dropout)
        self.mlp=MLP(p_dropout=dropout,vocab=vocab)
        self.embedding=nn.Embedding(num_embeddings=24,embedding_dim=64)
    
    def forward(self,batch):
        batch=batch["S"]
        seq_len=batch.size(1)
        batch=F.pad(batch, (0, 500-seq_len,0,0), mode='constant', value=0) # for some seq less than 500
        batch=self.embedding(batch)  #[32,500,64]
        cnn1=self.dccnn1(batch)
        cnn2=self.dccnn2(batch) # [32,500,16]
        cnn3=self.dccnn3(batch)
        bcl=self.bcl(cnn1,cnn2,cnn3)
        out=self.mlp(bcl) # [32,500,16]
        out=out[:,:seq_len,:]
        return {"log_probs":out}