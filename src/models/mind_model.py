import torch.nn as nn
from torch_geometric.data import  Data,Batch
from src.modules.mind_module import *


class MIND_model(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()
        self.p_dropout=args.dropout
        self.encoder_num=args.encoder_num
        self.output_dim=args.vocab
        self.embedding=Embedding()
        self.bilstm=BiLSTM()
        self.encoders=Encoders(length=self.encoder_num,dropout=self.p_dropout)
        self.graphattention=GraphAttention(dropout=self.p_dropout)
        self.mlp=MLP(input_dim=128,output_dim=self.output_dim)
        #self.mergemlp=MergeMLP(dropout=self.p_dropout)
        self.maskmlp=MaskMLP()
        #self.mergeattention=MergeAttention(dropout=self.p_dropout)
    def forward(self,batch): #[32:500]
        batch_s, batch_x = batch["S"], batch["X"]
        batch_len=len(batch_s)
        #self.encoders.to(device=device)
        seq_len=batch_s.size(1)
        batch_s1=F.pad(batch_s, (0, 500-seq_len), mode='constant', value=0)
        # branch 1
        embedded=self.embedding(batch_s1)  # [32,500,32]
        bilstm_out=self.bilstm(embedded)  # [32,500,32]
        x1=self.encoders(bilstm_out) # [32,500,64]

        # branch 2
        mask=self.maskmlp(bilstm_out)
        graph=MapConstruction(batch_x,mask)
        data_list=[]
        for i in graph:
            edge_index = torch.nonzero(i, as_tuple=False).t().to(batch_x.device)
            data=Data(i,edge_index)
            data_list.append(data)
        graph_batched=Batch.from_data_list(data_list)
        graphatt=self.graphattention(graph_batched) #[64000,64]
        graphatt = graphatt.view(batch_len, 2000, 64) # [32,2000,64]
        #x2 =self.mergeattention(x=graphatt,seq_info=bilstm_out)
        x2=graphatt
        #x2=self.mergemlp(bilstm_out,graphatt)
        x2=F.avg_pool1d(x2.permute(0, 2, 1),kernel_size=4).permute(0, 2, 1)  # [32,500,64]
        x_cat=torch.cat((x1,x2),dim=2) # [32,500,128]
        
        out=self.mlp(x_cat)
        out=out[:,:seq_len,:]
        return {'log_probs': out}