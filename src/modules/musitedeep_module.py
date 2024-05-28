import torch
import torch.nn as nn
class CNN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,dropout=0.75) -> None:
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.p_dropout=dropout
        self.kernel_size=kernel_size
        
        self.cnn=nn.Conv1d(in_channels=self.in_channels,out_channels=self.out_channels,kernel_size=self.kernel_size,padding = "same")
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(self.p_dropout)
        
    def forward(self,batch):
        cnn1=self.cnn(batch)
        cnn1=self.relu(cnn1)
        cnn1=self.dropout(cnn1)
        return cnn1


class MLP(nn.Module):
    def __init__(self,in_dim,out_dim,mid_dim=32,p_dropout=0.75) -> None:
        super().__init__()
        self.mlp1=nn.Linear(in_dim,mid_dim)
        self.mlp2=nn.Linear(mid_dim,mid_dim)
        self.mlp3=nn.Linear(mid_dim,out_dim)
        self.dropout=nn.Dropout(p_dropout)
        self.relu=nn.ReLU()
    def forward(self,batch):
        mlp1=self.mlp1(batch)
        mlp1=self.dropout(mlp1)
        mlp1=self.relu(mlp1)

        mlp2=self.mlp2(mlp1)
        mlp2=self.dropout(mlp2)
        mlp2=self.relu(mlp2)

        mlp3=self.mlp3(mlp2)
        mlp3=self.dropout(mlp3)
        mlp3=self.relu(mlp3)
        return mlp3   

class Attention(nn.Module):
    def __init__(self,embed_dim,p_dropout=0.75) -> None:
        super().__init__()
        self.embed_dim=embed_dim
        self.attention=nn.MultiheadAttention(embed_dim=self.embed_dim,num_heads=4,dropout=p_dropout)
    def forward(self,batch):
        output,_=self.attention(batch,batch,batch)
        return output
