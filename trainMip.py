# configuration
from config_mipnerf import get_config

# from mipnerf model import whole system
from mipnerf_system import MipNeRFSystem

# pytorch-lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

def main(hparams): # training main
    system = MipNeRFSystem(hparams) # build MipNERFsystem(class) with hp to system
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts_mip/{hparams.exp_name}',
                              filename='{step}',
                              monitor='val/psnr',
                              mode='max',   
                              save_top_k=-1,
                              every_n_train_steps=10000)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs_mip",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer=Trainer(max_steps=hparams.max_steps,
                    max_epochs=-1,
                    callbacks=callbacks,
                    logger=logger,
                    enable_model_summary=False,
                    accelerator='auto',
                    devices=1,
                    num_sanity_val_steps=1,
                    benchmark=True,
                    profiler="simple")

    trainer.fit(system)


if __name__ == "__main__":
    hparams = get_config()  # to hyper parameters to hparams
    main(hparams)
