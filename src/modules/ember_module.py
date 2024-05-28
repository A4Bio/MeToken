import numpy as np
import pandas as pd
from scipy.spatial import distance
from collections import Counter
import time

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim

# actually not in use
class SiameseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, z1, z2, label, margin=2.0, dist_type='L2'):
        if dist_type == 'L2':
            distance = F.pairwise_distance(z1, z2, p=2)
        elif dist_type == 'L1':
            distance = F.pairwise_distance(z1, z2, p=1)
        
        pair_losses = []
        for i in range(len(z1)):
            matrix_distance = distance[i]  # 直接使用预先计算好的距离
            pair_loss = torch.sum(label[i] * torch.pow(torch.clamp(margin - matrix_distance, min=0.0), 2))
            pair_losses.append(pair_loss)
        
        siam_loss = torch.mean(torch.stack(pair_losses))
        
        return siam_loss
    
class Siamese_embedding(nn.Module):
    def __init__(self,dropout=0.01):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=24, embedding_dim=8)   
        
        self.lstm = nn.LSTM(8, 8, bidirectional=True,num_layers=2,# [32,8*500*2]
                            batch_first=True,dropout=0.2)
        
        self.fc1 = nn.Linear(8000, 1000)
        self.fc2 = nn.Linear(1000, 256)
        #self.fc3 = nn.Linear(2048, 1024)
        #self.out = nn.Linear(1024, 1024)
        
        self.relu = nn.ReLU()
        self.drpt = nn.Dropout(p=dropout) #[32:256]
                
    def forward_once(self, motif): 
        batch_len=motif.size(0)
        motif=torch.unsqueeze(motif,dim=-1) # [32,500,1]
        motif_embedded=self.embedding(motif)
        motif_embedded=torch.squeeze(motif_embedded)
        lstmed, _ = self.lstm(motif_embedded.float()) # [32,500,16]
        if batch_len==1:
            lstmed=torch.unsqueeze(lstmed,0)
        flattened = torch.flatten(lstmed,1) 
    
        fc1 = self.fc1(flattened)
        fc1 = self.relu(fc1)
        fc1 = self.drpt(fc1)
        
        fc2 = self.fc2(fc1)
        fc2 = self.relu(fc2)
        #fc2 = self.drpt(fc2)
        
        #fc3 = self.fc3(fc2)
        #fc3 = self.relu(fc3)

        #out = self.out(fc3)

        return fc2
        
    def forward(self, batch):
        if batch.size(0)%2==0:
            split_point = batch.size(0) // 2
        else:
            split_point = batch.size(0) // 2 +1
        batch1,batch2=torch.split(batch,split_point)
        embed_1 = self.forward_once(batch1)
        embed_2 = self.forward_once(batch2)
        concatenated_tensor = torch.cat((embed_1, embed_2), dim=0)
        #return (embed_1, embed_2)
        return concatenated_tensor

class SequenceCNN(nn.Module):
    def __init__(self,conv_drpt=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(500, 250, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.penult = nn.Linear(250, 32)
        self.relu = nn.ReLU()
        self.conv_drpt = nn.Dropout(p = conv_drpt)
        self.ablate = nn.Dropout(p = 1.0)

    def forward(self,seq):
        seq_unsqueezed=torch.unsqueeze(seq,dim=-1)
        conv1 = self.conv1(seq_unsqueezed.float())
        conv1 = self.relu(conv1)
        #conv1 = self.pool(conv1)
        #conv2 = self.conv2(conv1)
        #conv2 = self.relu(conv2)
        #conv2 = self.pool(conv2)
        #conv3 = self.conv3(conv2)
        conv3 = self.relu(conv1)
        #conv3 = self.pool(conv3)
        seq_out = conv3.view(conv3.size()[0], -1)
        seq_out = self.penult(seq_out) ## SEQ PENULT
        seq_out = self.relu(seq_out)
        seq_out = self.conv_drpt(seq_out)
        return seq_out

class Coord_MLP(nn.Module):
    def __init__(self,mlp_drpt=0.0):
        super().__init__()
        self.relu = nn.ReLU()
        self.mlp1 = nn.Linear(256, 64)
        self.bn1 = nn.BatchNorm1d(64)
#        self.mlp2 = nn.Linear(1024, 1024)
#        self.bn2 = nn.BatchNorm1d(1024)
        self.mlp3 = nn.Linear(64, 32)
#        self.bn3 = nn.BatchNorm1d(500)
#        self.penult = nn.Linear(500, 250)
        self.mlp_drpt = nn.Dropout(p = mlp_drpt)
    
    def forward(self,coords):
        mlp1 = self.mlp1(coords)
        mlp1 = self.relu(mlp1)
        mlp1 = self.bn1(mlp1)
        mlp1 = self.mlp_drpt(mlp1)
        #mlp2 = self.mlp2(mlp1)
        #mlp2 = self.relu(mlp2)
        #mlp2 = self.bn2(mlp2)
        #mlp2 = self.mlp_drpt(mlp2)
        mlp3 = self.mlp3(mlp1)
        mlp3 = self.relu(mlp3)
#        mlp3 = self.bn3(mlp3)
#        mlp3 = self.mlp_drpt(mlp3)
#        coord_out = self.penult(mlp3)
        coord_out = self.relu(mlp3)
        return coord_out

class Concat_MLP(nn.Module):
    def __init__(self,vocabs):
        super().__init__()
        self.vocabs=vocabs
        self.penult = nn.Linear(64, 256)#[32:64]
        self.out = nn.Linear(256, 500*vocabs)
    
    def forward(self,seq_out,coords_out):
        cat = torch.cat((seq_out,coords_out), 1)
        penult=self.penult(cat)
        out=self.out(penult)
        out_3d = torch.reshape(out, (len(seq_out), 500, self.vocabs))
        return out_3d