import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, EsmModel


class ESM_Model(nn.Module):
    def __init__(self, args):
        """ Graph labeling network """
        super(ESM_Model, self).__init__()
        self.args=args
        # {0: '<cls>', 1: '<pad>', 2: '<eos>', 3: '<unk>', 4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T', 12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y', 20: 'M', 21: 'H', 22: 'W', 23: 'C', 24: 'X', 25: 'B', 26: 'U', 27: 'Z', 28: 'O', 29: '.', 30: '-', 31: '<null_1>', 32: '<mask>'}
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/tancheng/model_zoom/transformers")
        self.esm = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir="/tancheng/model_zoom/transformers")
        for param in self.esm.parameters():
            param.requires_grad = False

        hidden_dim = self.esm.pooler.dense.in_features
        self.ptm_predicter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, args.vocab)
        )
        
    def forward(self, batch):
        outputs = self.esm(input_ids=batch['S'])
        logits = self.ptm_predicter(outputs.last_hidden_state)
        log_probs = F.log_softmax(logits, dim=-1)
        return {'log_probs': log_probs}