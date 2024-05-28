import torch
import torch.nn as nn
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim,output_dim=64,p_dropout=0.4) -> None:
        super().__init__()
        self.embedding_dim=embedding_dim
        self.output_dim=output_dim
        self.lstm=nn.LSTM(self.embedding_dim, self.output_dim, bidirectional=True,num_layers=2,
                            batch_first=True,dropout=p_dropout)
        self.timedistributed=nn.Linear(self.output_dim*2,self.output_dim)
        self.relu=nn.ReLU()
    def forward(self,batch):
        batch,_=self.lstm(batch)
        batch=self.timedistributed(batch)
        batch=self.relu(batch)
        return batch

class MLP(nn.Module):
    def __init__(self, in_dim,out_dim,mid_dim=4,channel=500,p_dropout=0.4) -> None:
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.mid_dim=mid_dim
        self.channel=channel
        self.mlp1=nn.Linear(self.in_dim*channel,self.mid_dim*channel)
        self.dropout=nn.Dropout(p_dropout)
        self.relu=nn.ReLU()
        self.mlp2=nn.Linear(self.mid_dim,self.out_dim)
        
    def forward(self,batch):
        batch_size=batch.shape[0]
        batch=batch.flatten(start_dim=1)
        mlp1=self.mlp1(batch)
        mlp1=self.dropout(mlp1)
        mlp1=self.relu(mlp1)
        mlp1=mlp1.reshape(batch_size,self.channel,self.mid_dim)
        mlp2=self.mlp2(mlp1)
        mlp2=self.dropout(mlp2)
        mlp2=self.relu(mlp2)
        return mlp2