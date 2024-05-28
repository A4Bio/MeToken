import torch.nn as nn
import torch.nn.functional as F
from src.modules.ember_module import *


class EMBER_model(nn.Module):
    def __init__(self, args, **kwargs):
        """ Graph labeling network """
        super().__init__()
        self.args = args
        mlp_dropout = args.mlp_dropout
        conv_dropout=args.conv_dropout
        siamese_dropout=args.siamese_dropout
        self.vocab=args.vocab
        self.siamese_embedding=Siamese_embedding(dropout=siamese_dropout)
        self.seqcnn=SequenceCNN(conv_drpt=conv_dropout)
        self.coord_mlp=Coord_MLP(mlp_drpt=mlp_dropout)
        self.concat_mlp=Concat_MLP(self.vocab)

    def forward(self, batch):
        batch=batch["S"]
        seq_len=batch.size(1)
        batch=F.pad(batch, (0, 500-seq_len), mode='constant', value=0)
        # PATH 1
        embedded=self.siamese_embedding.forward_once(batch)
        mlped_1=self.coord_mlp(embedded)
        
        # PATH 2
        cnn_result=self.seqcnn(batch)

        # RESULT
        logits=self.concat_mlp(mlped_1,cnn_result)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs=log_probs[:,:seq_len,:]
        # return log_probs, log_probs0
        return {'log_probs': log_probs}


    
