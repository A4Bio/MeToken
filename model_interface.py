import inspect
import pickle
from unittest import result
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import torch
import torch.nn as nn
import os
import sys
import wandb
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.interface.model_interface import MInterface_base
import math
from omegaconf import OmegaConf
from torchmetrics import AUROC
from torcheval.metrics import MulticlassAUPRC
from torch.autograd import Variable
import torch.nn.functional as F


class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.load_model()

        self.test_idxs = []
        self.test_tokenemb=[]
        self.test_nodeemb=[]
        self.test_lastemb=[]
        self.test_predemb=[]
        self.codebook=[]

        self.test_preds = []
        self.test_probs = []
        self.test_trues = []
        self.vocab=26
        self.loss_function = FocalLoss(gamma=5.0)
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

    def init_loss_function(self, gamma, weight_type):
        self.loss_function = FocalLoss(gamma=gamma, weight_type=weight_type)

    def to_onehot(self, labels):
        return np.eyes(self.hparams.vocab)[labels]

    def forward(self, batch, mode='train', temperature=1.0):
        with open("./results/output.txt","w")as out:
            out.write(str(batch))
        if self.hparams.augment_eps>0:
            batch['X'] = batch['X'] + self.hparams.augment_eps * torch.randn_like(batch['X'])

        results = self.model(batch)
        valid_idx = batch['Q'] > 0 if self.hparams.with_null_ptm == 0 else torch.ones_like(batch['Q'])
        gt_ptm, log_probs, mask = batch['Q'][valid_idx], results['log_probs'][valid_idx], batch['mask'][valid_idx] # valid_idx should be [5878]
        if len(log_probs.shape) == 3:
            loss = self.loss_function(log_probs.permute(0,2,1), gt_ptm)
            loss = (loss*mask).sum()/(mask.sum())
        elif len(log_probs.shape) == 2:
            loss = self.loss_function(log_probs, gt_ptm)
            loss = (loss*mask).sum()/(mask.sum())
        preds = log_probs.argmax(dim=-1)[mask == 1.].cpu().tolist()
        probs = log_probs.softmax(dim=-1)[mask == 1.].cpu().tolist()
        trues = gt_ptm[mask == 1.].cpu().tolist()
        #self.test_nodeemb.append(results["node_emb"])
        #self.test_tokenemb.append(results["token_emb"])
        #self.test_predemb.append(results["pred_emb"])
        #self.test_lastemb.append(results["last_emb"])
        self.test_idxs.append(results["token_index"])
        #self.codebook=results["codebook"]
        return loss, preds, probs, trues

    def training_step(self, batch, batch_idx, **kwargs):
        loss,  preds, probs, trues = self(batch)
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss,  preds, probs, trues = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'preds': preds, 'labels': trues}

    def validation_epoch_end(self, outputs):
        preds = np.concatenate([x['preds'] for x in outputs])
        labels = np.concatenate([x['labels'] for x in outputs])
        f1_macro = f1_score(preds, labels, average='macro')
        self.log('val_f1', f1_macro, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, preds, probs, trues = self(batch)
        self.test_idxs.extend(batch['id'])
        self.test_preds.extend(preds)
        self.test_probs.extend(probs)
        self.test_trues.extend(trues)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self,batch,*args, **kwargs):
        ori_labels=batch["Q"]
        length_list=batch["lengths"]
        _,preds,_,_=self(batch)
        preds_dict=[]
        for i,j in enumerate(ori_labels):
            non_zero_indices = torch.nonzero(j, as_tuple=True)[0]
            result_dict = {index.item(): preds[i] for i, index in enumerate(non_zero_indices)} # index problem need to be fixed
            preds_dict.append(result_dict)
        return preds_dict

    def cal_metric(self, path):
        preds, probs, trues = np.array(self.test_preds), np.array(self.test_probs), np.array(self.test_trues)

        if 'generalization' in path:
            classes = [1, 10, 16, 23, 24]
            cls_idx = np.isin(trues, classes)
            preds, probs, trues = preds[cls_idx], probs[cls_idx], trues[cls_idx]

        accuracy = accuracy_score(trues, preds)
        recall = recall_score(trues, preds,average="macro")
        mcc = matthews_corrcoef(trues, preds)
        precision_macro = precision_score(trues, preds, average='macro')
        f1_macro = f1_score(trues, preds, average='macro')

        if 'generalization' not in path:
            # auroc
            vocab = self.vocab if self.hparams.with_null_ptm == 1 else self.vocab - 1
            stat_trues = trues if self.hparams.with_null_ptm == 1 else trues - 1
            stat_probs = probs if self.hparams.with_null_ptm == 1 else probs[:, 1:]
            roc_metric = AUROC(task="multiclass", num_classes=vocab)
            auroc = roc_metric(torch.tensor(stat_probs), torch.tensor(stat_trues))
            # auprc
            pr_metric = MulticlassAUPRC(num_classes=vocab)
            pr_metric.update(torch.tensor(stat_probs),torch.tensor(stat_trues))
            auprc = pr_metric.compute().item()
        else:
            vocab = len(classes)
            new_index = {v: i for i, v in enumerate(classes)}
            new_trues = np.vectorize(new_index.get)(trues)
            new_probs = probs[:, classes]

            roc_metric = AUROC(task="multiclass", num_classes=vocab)
            auroc = roc_metric(torch.tensor(new_probs), torch.tensor(new_trues))

            pr_metric = MulticlassAUPRC(num_classes=vocab)
            pr_metric.update(torch.tensor(new_probs), torch.tensor(new_trues))
            auprc = pr_metric.compute().item()

        if self.trainer.is_global_zero:
            wandb.log({
                'Test/Accuracy': accuracy,
                'Test/Precision': precision_macro,
                'Test/Recall': recall,
                'Test/F1-score': f1_macro,
                'Test/Mcc-score': mcc,
                'Test/AUROC': auroc,
                'Test/AUPRC': auprc
            })
        print(f'accuracy: {accuracy:.4f}, precision: {precision_macro:.4f}, recall: {recall:.4f}, f1 score: {f1_macro:.4f}, mcc score: {mcc:.4f}, auroc: {auroc:.4f}, auprc: {auprc:.4f}')
        return {'accuracy': accuracy, 'precision': precision_macro, 'recall': recall, 'f1_score': f1_macro, 'mcc_score': mcc, 'auroc': auroc.item(), 'auprc': auprc}       
        
    def load_model(self):
        try:
            params = OmegaConf.load(f'./configs/{self.hparams.model_name}.yaml')
            params.update(self.hparams)
        except:
            params=None
        if self.hparams.model_name == 'PiFold':
            from src.models.pifold_model import PiFold_Model
            self.model = PiFold_Model(params)
        
        if self.hparams.model_name == "ESM":
            from src.models.esm_model import ESM_Model
            self.model = ESM_Model(params)

        if self.hparams.model_name == "EMBER":
            from src.models.ember_model import EMBER_model
            self.model = EMBER_model(params)

        if self.hparams.model_name == "MIND":
            from src.models.mind_model import MIND_model
            self.model = MIND_model(params)

        if self.hparams.model_name == "DeepPhos":
            from src.models.deepphos_model import DeepPhos_model
            self.model = DeepPhos_model(params)

        if self.hparams.model_name == 'MeToken':
            from src.models.metoken_model import MeToken_Model
            params.using_metoken = False
            self.model = MeToken_Model(params)

        if self.hparams.model_name == 'MeTokenv2':
            from src.models.metoken_model import MeToken_Model
            params.using_metoken = True
            self.model = MeToken_Model(params)

        if self.hparams.model_name == 'MeTokenPro':
            from src.models.metokenpro_model import MeTokenPro_Model
            params.using_metoken = True
            self.model = MeTokenPro_Model(params)
        
        if self.hparams.model_name == 'MeTokenMax':
            from src.models.metokenmax_model import MeTokenMax_Model
            params.using_metoken = True
            self.model = MeTokenMax_Model(params)

        if self.hparams.model_name == 'StructGNN':
            from src.models.structgnn_model import StructGNN_Model
            self.model = StructGNN_Model(params)
        
        if self.hparams.model_name == 'GraphTrans':
            from src.models.graphtrans_model import GraphTrans_Model
            self.model = GraphTrans_Model(params)

        if self.hparams.model_name == 'GVP':
            from src.models.gvp_model import GVP_Model
            self.model = GVP_Model(params)

        if self.hparams.model_name == 'CDConv':
            from src.models.cdconv_model import CDConv_Model
            self.model = CDConv_Model(params)

        if self.hparams.model_name == 'MusiteDeep':
            from src.models.musitedeep_model import MusiteDeep_Model
            self.model = MusiteDeep_Model(params)

        if self.hparams.model_name == 'DTL_DephosSite':
            from src.models.dtl_dephossite_model import DTL_DephosSite_Model
            self.model = DTL_DephosSite_Model(params)


    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight_type=0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        class_samples = [106162623, 104297, 2173, 207, 114, 443, 2689, 1948, 273, 3538, 9751, 204, 878, 15332, 7549, 360, 1098429, 638, 2479, 4147, 6630, 169, 4090, 2045, 164054, 1098]
        total_samples = sum(class_samples)
        weight = torch.tensor([total_samples / x for x in class_samples])
        if weight_type == 0:
            self.weight = None
        elif weight_type == 1:
            self.weight = weight / weight.min()
        elif weight_type == 2:
            self.weight = weight / weight.max()

    def forward(self, log_prob, target):
        prob = torch.exp(log_prob)
        targets_one_hot = F.one_hot(target, num_classes=log_prob.size(-1))

        prob = torch.sum(prob * targets_one_hot, dim=-1)
        log_prob = torch.sum(log_prob * targets_one_hot, dim=-1)

        focal_loss = -1 * (1 - prob) ** self.gamma * log_prob

        if self.weight is not None:
            self.weight = self.weight.type_as(log_prob)
            focal_loss *= self.weight[target]

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss