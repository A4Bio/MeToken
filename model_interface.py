import inspect
import torch
import os
import numpy as np
from omegaconf import OmegaConf
from torchmetrics import AUROC
from torcheval.metrics import MulticlassAUPRC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
import pytorch_lightning as pl
from src.metoken_model import MeToken_Model


class MInterface(pl.LightningModule):
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
        self.vocab = 26
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

    def to_onehot(self, labels):
        return np.eyes(self.hparams.vocab)[labels]

    def forward(self, batch, mode='train', temperature=1.0):
        with open("./results/output.txt","w")as out:
            out.write(str(batch))
        if self.hparams.augment_eps>0:
            batch['X'] = batch['X'] + self.hparams.augment_eps * torch.randn_like(batch['X'])

        results = self.model(batch)
        valid_idx = batch['Q'] > 0 if self.hparams.with_null_ptm == 0 else torch.ones_like(batch['Q'])
        gt_ptm, log_probs, mask = batch['Q'][valid_idx], results['log_probs'][valid_idx], batch['mask'][valid_idx]

        preds = log_probs.argmax(dim=-1)[mask == 1.].cpu().tolist()
        probs = log_probs.softmax(dim=-1)[mask == 1.].cpu().tolist()
        trues = gt_ptm[mask == 1.].cpu().tolist()
        self.test_idxs.append(results["token_index"])
        return preds, probs, trues
    
    def test_step(self, batch, batch_idx):
        preds, probs, trues = self(batch)
        self.test_idxs.extend(batch['id'])
        self.test_preds.extend(preds)
        self.test_probs.extend(probs)
        self.test_trues.extend(trues)
        return 
    
    def predict_step(self, batch, *args, **kwargs):
        ori_labels = batch["Q"]
        preds, _, _ = self(batch)
        preds_dict=[]
        for i,j in enumerate(ori_labels):
            non_zero_indices = torch.nonzero(j, as_tuple=True)[0]
            result_dict = {index.item(): preds[i] for i, index in enumerate(non_zero_indices)}
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

        print(f'accuracy: {accuracy:.4f}, precision: {precision_macro:.4f}, recall: {recall:.4f}, f1 score: {f1_macro:.4f}, mcc score: {mcc:.4f}, auroc: {auroc:.4f}, auprc: {auprc:.4f}')
        return {'accuracy': accuracy, 'precision': precision_macro, 'recall': recall, 'f1_score': f1_macro, 'mcc_score': mcc, 'auroc': auroc.item(), 'auprc': auprc}       
        
    def load_model(self):
        try:
            params = OmegaConf.load(f'./configs/{self.hparams.model_name}.yaml')
            params.update(self.hparams)
        except:
            params = None
        
        if self.hparams.model_name == 'MeToken':
            params.using_metoken = True
            self.model = MeToken_Model(params)

    def instancialize(self, Model, **other_args):
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)