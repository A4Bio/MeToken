import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch

class Embedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=24,embedding_dim=32)
    
    def forward(self,batch): #[32,500]
        embedded=self.embedding(batch)
        return embedded

class BiLSTM(nn.Module):
    def __init__(self,embedding_dim=500,output_dim=250) -> None:
        super().__init__()
        self.embedding_dim=embedding_dim
        self.output_dim=output_dim
        self.lstm = nn.LSTM(self.embedding_dim, self.output_dim, bidirectional=True,num_layers=2,
                            batch_first=True,dropout=0.2)
        self.fc=nn.Linear(32,64)
    
    def forward(self,batch): 
        batch=torch.permute(batch,(0,2,1))
        output,lstmout=self.lstm(batch)
        output=torch.permute(output,(0,2,1))
        output=self.fc(output)
        return output
    
def MapConstruction(tensor, mask,threshold=2):
    batch_size = tensor.size(0)
    adjacency_matrices = []
    for i in range(batch_size):
        coordinates = tensor[i].view(-1, 3)  # 将坐标展平为500*4个原子的三维坐标

        # 计算距离矩阵
        diff = coordinates[:, None, :] - coordinates
        distances = torch.norm(diff, dim=2)

        # 根据阈值生成邻接矩阵
        adjacency_matrix = (distances < threshold).float()
        extended_adjacency_matrix = torch.zeros(2000, 2000).to(tensor.device)
        extended_adjacency_matrix[:adjacency_matrix.shape[0], :adjacency_matrix.shape[1]] = adjacency_matrix
        extended_adjacency_matrix*=mask[i]
        extended_adjacency_matrix*=mask[i].T
        adjacency_matrices.append(extended_adjacency_matrix)

    return torch.stack(adjacency_matrices)

class Encoder(nn.Module):
    def __init__(self,p_droupot) -> None:
        super().__init__()
        self.dropout=nn.Dropout(p_droupot)
        self.fc1=nn.Linear(64,64)
        self.layernorm=nn.LayerNorm(64)
        self.attention=nn.MultiheadAttention(embed_dim=64, num_heads=4)
        self.relu=nn.ReLU()

    def forward(self,batch):
        batch=batch

        batch_unsqueezed = torch.permute(batch,(1,0,2))
        outputs, att_weights = self.attention(batch_unsqueezed,batch_unsqueezed,batch_unsqueezed) # [500,32,32]
        att=torch.permute(outputs,(1,0,2))

        att=self.dropout(att)
        att=self.layernorm(att+batch) #resnet-like batch and att [32,500,64] 
        fc=self.fc1(att)
        fc=self.relu(fc)
        fc=self.dropout(fc)
        out=self.layernorm(fc+att) # same
        return out

class Encoders(nn.Module):
    def __init__(self,length,dropout) -> None:
        super().__init__()
        self.encoder_num=length
        self.p_droupot=dropout
        self.encoders=nn.Sequential(*[Encoder(self.p_droupot) for self.encoder_num in range(self.encoder_num)])
    def forward(self,batch):
        out=batch # [32,500,16]
        for i in self.encoders:
            out=i(out)
        return out

class GraphAttention(nn.Module):
    def __init__(self, in_features=2000, hidden_dim=16, out_features=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_features, out_channels=hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, out_features, dropout=dropout)
        self.dropout=nn.Dropout(p=self.dropout)
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm1d(out_features)
        self.bn2=nn.BatchNorm1d(out_features)

    def forward(self, x):
        conv1 = self.conv1(x.x,x.edge_index)
        conv1=self.bn1(conv1)
        conv1 =  self.dropout(conv1)
        conv2 = self.conv2(conv1,x.edge_index)
        conv2=self.bn2(conv2)
        return conv2

class MLP(nn.Module):
    def __init__(self,input_dim=128,p_dropout=0.1,output_dim=26)-> None:
        super().__init__()
        self.fc1=nn.Linear(input_dim,100)
        self.dropout=nn.Dropout(p_dropout)
        self.bn=nn.BatchNorm1d(500)
        self.fc3=nn.Linear(100,output_dim)
        self.relu=nn.ReLU()
    def forward(self,batch): #[32:500]
        fc1=self.fc1(batch)
        fc1=self.relu(fc1)
        fc1=self.dropout(fc1)
        fc1=self.bn(fc1)
        fc3=self.fc3(fc1)
        fc3=self.relu(fc3)
        fc3=self.dropout(fc3)
        fc3=self.bn(fc3)
        out=F.log_softmax(fc3,dim=-1)
        return out

def adjacency_matrix_to_edge_index(adjacency_matrix):
    edge_index = []

    for i in range(adjacency_matrix.size(0)):
        for j in range(i + 1, adjacency_matrix.size(1)):
            if adjacency_matrix[i, j] == 1:
                edge_index.append([i, j])

    return torch.tensor(edge_index).t().contiguous()

class MergeAttention(nn.Module):
    def __init__(self, in_features=64, num_heads=4, dropout=0.1):
        super().__init__() # [32,2000,64] [32,500,64]
        self.dropout = dropout
        self.fc1=nn.Linear(500*64,40)
        self.fc2=nn.Linear(40,2000*64)
        self.bn1=nn.BatchNorm1d(40)
        self.bn2=nn.BatchNorm1d(2000*64)
        self.att=nn.MultiheadAttention(embed_dim=in_features,num_heads=num_heads,dropout=dropout)
        self.dropout=nn.Dropout(p=self.dropout)
        self.relu=nn.ReLU()
        self.cuda()

    def forward(self, x, seq_info):
        seq_info=seq_info.reshape(len(x),500*64)
        seq_mask=self.fc1(seq_info)
        seq_mask=self.bn1(seq_mask)
        seq_mask=self.relu(seq_mask)

        seq_mask=self.fc2(seq_mask)
        seq_mask=self.bn2(seq_mask)
        seq_mask=self.relu(seq_mask)
        seq_mask=seq_mask.reshape(len(x),2000,64)
        x=torch.permute(x,(1,0,2))
        seq_mask=torch.permute(seq_mask,(1,0,2))
        att,att_weights=self.att(seq_mask,x,x)
        att=torch.permute(att,(1,0,2))
        out=att
        return out

class MergeMLP(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__() # [32,2000,64] [32,500,64]
        self.fc1=nn.Linear(2500*64,40)
        self.fc2=nn.Linear(40,500*64)
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm1d(40)
        self.bn2=nn.BatchNorm1d(500*64)
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,lstm,x2):
        x_fc1=torch.cat((lstm,x2),dim=1)
        x_fc1=x_fc1.reshape(len(x2),2500*64)
        x_fc1=self.fc1(x_fc1)
        x_fc1=self.bn1(x_fc1)
        x_fc1=self.relu(x_fc1)
        x_fc2=self.fc2(x_fc1)
        x_fc2=self.bn2(x_fc2)
        x_fc2=self.relu(x_fc2)
        out=x_fc2.reshape(len(x2),500,64)
        return out
    
class MaskMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(500*64,100)
        self.fc2=nn.Linear(100,2000)
        self.sigmoid=nn.Sigmoid()
        self.relu=nn.ReLU()
        
    def forward(self,batch):
        batch=torch.reshape(batch,(len(batch),500*64))
        fc1=self.fc1(batch)
        fc1=self.relu(fc1)
        fc2=self.fc2(fc1)
        output=self.sigmoid(fc2)
        output = torch.where(output < 0.5, torch.tensor(0), torch.tensor(1))
        return output
    
