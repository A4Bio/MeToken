import os
import os.path as osp
import shutil
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint


def clean_temp_files(tmp_dir='/tmp'):
    for item in os.listdir(tmp_dir):
        item_path = os.path.join(tmp_dir, item)
        if os.path.isdir(item_path) and item.startswith('pymp'):
            try:
                shutil.rmtree(item_path)
            except Exception as e:
                pass

class SetupCallback(Callback):
    def __init__(self,  now, logdir, ckptdir, cfgdir, config, argv_content=None):
        super().__init__()
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
    
        self.argv_content = argv_content

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                            os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))
        
            with open(os.path.join(self.logdir, "argv_content.txt"), "w") as f:
                f.write(str(self.argv_content))

class BackupCodeCallback(Callback):
    def __init__(self, source_dir, backup_dir, ignore_patterns=None):
        super().__init__()
        self.source_dir = source_dir
        self.backup_dir = backup_dir
        self.ignore_patterns = ignore_patterns

    def on_train_start(self, trainer, pl_module):
        try:
            if trainer.global_rank == 0:
                os.makedirs(self.backup_dir, exist_ok=True)
                if os.path.exists(self.backup_dir+'/code'):
                    shutil.rmtree(self.backup_dir+'/code')
                shutil.copytree(self.source_dir, self.backup_dir+'/code', ignore=self.ignore_patterns)

                print(f"Code file backed up to {self.backup_dir}")
        except:
            print(f"Fail in copying file backed up to {self.backup_dir}")


class TempFileCleanupCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_counter = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        self.epoch_counter += 1
        if self.epoch_counter % 5 == 0:
            if trainer.global_rank == 0:
                # clean_temp_files("/tmp")
                pass


class BestCheckpointCallback(ModelCheckpoint):
    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))

    def on_test_end(self, trainer, pl_module):
        super().on_test_end(trainer, pl_module)
        checkpoint_callback = trainer.checkpoint_callback
        if checkpoint_callback and checkpoint_callback.best_model_path and trainer.global_rank == 0:
            best_path = checkpoint_callback.best_model_path
            shutil.copy(best_path, osp.join(osp.dirname(best_path), 'best.ckpt'))