import inspect
from torch.utils.data import DataLoader
from src.interface.data_interface import DInterface_base


class DInterface(DInterface_base):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.load_data_module()

    def setup(self, stage=None):
        from src.datasets.featurizer import featurize_GTrans,featurize_GVP
        # if self.hparams.model_name in ['PiFold']:
        if self.hparams.model_name=="GVP":
            self.collate_fn_class=featurize_GVP()
            self.collate_fn = self.collate_fn_class.collate
        else:
            self.collate_fn = featurize_GTrans

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(split = 'train')
            self.valset = self.instancialize(split='valid')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(split='test')
        # prediction only
        if stage == "predict":
            self.predictset=self.instancialize(split='predict')

        if stage == 'test' :
            self.test_loader = self.test_dataloader()
        elif stage=="predict":
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
        name = self.hparams.dataset
        if name=="PTM":
            from src.datasets.ptm_dataset import PTMDataset
            self.data_module=PTMDataset

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        
        class_args =  list(inspect.signature(self.data_module.__init__).parameters)[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.hparams[arg]
        args1.update(other_args)
        return self.data_module(**args1)