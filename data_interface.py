import inspect
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.datasets.ptm_dataset import PTMDataset
from src.datasets.featurizer import featurize


class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = self.hparams.batch_size
        self.load_data_module()

    def setup(self, stage=None):
        self.collate_fn = featurize

        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(split='train')
            self.valset = self.instancialize(split='valid')
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(split='test')
        if stage == "predict":
            self.predictset = self.instancialize(split='predict')

        if stage == 'test' :
            self.test_loader = self.test_dataloader()
        elif stage == "predict":
            self.predict_loader=self.predict_dataloader()
        else:
            self.train_loader = self.train_dataloader()
            self.val_loader = self.val_dataloader()
            self.test_loader = self.test_dataloader()

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.hparams.num_workers, shuffle=True, pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.predictset, batch_size=self.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True, collate_fn=self.collate_fn)
    
    def load_data_module(self):
        self.data_module = PTMDataset

    def instancialize(self, **other_args):
        class_args = list(inspect.signature(self.data_module.__init__).parameters)[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.hparams[arg]
        args1.update(other_args)
        return self.data_module(**args1)