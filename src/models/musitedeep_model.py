import torch.nn as nn
from torch_geometric.data import  Data,Batch
import torch.nn.functional as F
from src.modules.musitedeep_module import *

class MusiteDeep_Model(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.in_channel=args.in_channel # second dim
        self.mid_channel=args.mid_channel
        self.out_channel=args.out_channel
        self.in_dim=args.in_dim # third dim
        self.mid_dim=args.mid_dim
        self.out_dim=args.vocab
        self.merge_middim=256
        self.embedding=nn.Embedding(num_embeddings=24,embedding_dim=24)
        self.cnn1=CNN(self.in_channel,self.mid_channel,1)
        self.cnn2=CNN(self.mid_channel,self.mid_channel,9)
        self.cnn3=CNN(self.mid_channel,self.out_channel,10)
        self.attention1=Attention(self.in_dim)
        self.attention2=Attention(self.out_channel)
        self.merged_mlp=MLP(self.in_dim*self.out_channel*2,self.mid_dim*self.in_channel,self.merge_middim)
        self.mlp=MLP(self.mid_dim,self.out_dim)
    def forward(self,batch):
        batch=batch["S"] # [16,500]
        batch_len = batch.shape[1]
        batch=F.pad(batch, (0, 500-batch_len), "constant", 0)
        batchlen=batch.shape[0]
        embedded=self.embedding(batch)# [16,500,24]
        cnn=self.cnn1(embedded)
        cnn=self.cnn2(cnn)
        cnn=self.cnn3(cnn) # [16,64,24]
        attention1=cnn
        attention2=torch.transpose(cnn,1,2) # [16,24,64]
        attention1=self.attention1(attention1) # [16,64,24]
        attention1=attention1.reshape(batchlen,-1)
        attention2=self.attention2(attention2) # [16,24,64]
        attention2=attention2.reshape(batchlen,-1)        
        cated=torch.cat((attention1,attention2),-1) # [16,64*24*2]
        merged=self.merged_mlp(cated) # [16,24*64*2]-> [16,500*16]
        merged=merged.reshape(batchlen,self.in_channel,self.mid_dim)
        output=self.mlp(merged) # [16,500,16]->[16,500,26] 
        log_probs = F.log_softmax(output, dim=-1)[:, :batch_len]
        return {'log_probs': log_probs}

