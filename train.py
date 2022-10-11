# configuration
from config import get_config

# from nerf model import whole system
from nerf import NeRFSystem

# pytorch-lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.accelerators import accelerator

def main(hparams):  # training main
    system = NeRFSystem(hparams) # build NERFsystem(class) with hp to system
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{step}', # has a default 
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=-1)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='auto',
                      devices=1,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple")

    trainer.fit(system)

if __name__ == '__main__':      #journey begins..
    hparams = get_config()  # to hyper parameters to hparams
    main(hparams) # put hparams to main 