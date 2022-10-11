# modules 
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from collections import defaultdict
from pytorch_lightning import LightningModule

# loss
from loss import ColorLoss
# load dataset
from load_dataset import LLFFDataset
# from rendering model import renderrays
from rendering import render_rays
# for torch 
from model_utils import *
from metrics import *

class Embedding(nn.Module):
    def __init__(self, k):  
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        k : embedding freq
        eg.
            k = 3 embeds a scalar to:
            (x, sin(x), cos(x), sin(2x), cos(2x), sin(4x), cos(4x))
        """
        super().__init__()
        self.k = k              
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2**torch.linspace(0, k-1, k)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        add original x to the embedding!

        Inputs:
            x: (B, x)

        Outputs:
            out: (B, x+2*x*k) = (B,x(2k+1))

        usually: x should be dim 3 (xyz and dir)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)

class NeRF(nn.Module):  #MLP

    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+2*3*10=63 by default)
        in_channels_dir: number of input channels for direction (3+2*3*4=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        for i in range(D):
            if i == 0:  # original input
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)  # build model as name self.xyz_encoding_i (1-8)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),  # int
                                nn.ReLU(True))  # change input

        # output layers
        self.sigma = nn.Sequential(
                                nn.Linear(W, 1),
                                nn.ReLU(True))

        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).

        Inputs:
            x = (B, self.in_channels_xyz+self.in_channels_dir)
               the embedded vector of position and direction

        Outputs:
            (B, 4), rgb and sigma
        """

        #getting xyz and dir
        input_xyz, input_dir = \
            torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        xyz_ = input_xyz # original input 

        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1) # skip
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)  

        sigma = self.sigma(xyz_)  # output

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding) #output

        out = torch.cat([rgb, sigma], -1)

        return out

class NeRFSystem(LightningModule):  # subclass of lightningmodule
    def __init__(self, hparams):    #init with hp
        super().__init__()      
        self.save_hyperparameters(hparams)  # lightning feature, save hp, build in name is also hparams
        self.loss = ColorLoss() # to loss.py, define loss, self.loss is a nn.Module  #  when using, loss(x,y)
        self.embedding_xyz = Embedding(hparams.N_emb_xyz) # embedding is a nn class, emb_xyz: freq of pos encoding
        self.embedding_dir = Embedding(hparams.N_emb_dir) # embedding is a nn class, emb_dir: freq of pos encoding
            #   when using, testing embedding_xyz(x) (x : vector)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir} 

        in_channel_xyz = 3+2*3*hparams.N_emb_xyz  # for clarity : original xyz(3) + cos/sin(2)*xyz(3)*freq(N)
        in_channel_dir = 3+2*3*hparams.N_emb_dir  # for clarity : original dir(3) + cos/sin(2)*dir(3)*freq(N)
        
        # model coarse
        self.nerf_coarse = NeRF(in_channels_xyz=in_channel_xyz, # building model
                                in_channels_dir=in_channel_dir)
        self.models = {'coarse': self.nerf_coarse}  # a model directly output!

        # model fine if importance sampling > 0
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(in_channels_xyz=in_channel_xyz,
                                  in_channels_dir=in_channel_dir)
            self.models['fine'] = self.nerf_fine


    def forward(self, rays):    # this is a lightning module, so forward too, with rays
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]  # how many rays how many batches  (B,8)->B
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):  # chunks in batches, why B is smaller than chunk
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],   # per chunk, chunk size to split the input to avoid OOM, default 32*1024
                            self.hparams.N_samples,         # num of coarse samples, default 64
                            self.hparams.use_disp,          # default false,
                            self.hparams.perturb,           # default 1
                            self.hparams.noise_std,         # std dev of noise added to regularize sigma
                            self.hparams.N_importance,      # num of fine samples, default 128
                            self.hparams.chunk) # chunk size is effective in val mode, default 1024*32

            for k, v in rendered_ray_chunks.items():  # must returning a dict
                results[k] += [v]

        for k, v in results.items(): # whole batch finished
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):     # from lightning, set up dataset
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh),
                  'spheric_poses':self.hparams.spheric_poses}  # why 504 378, the output maybe
        self.train_dataset = LLFFDataset(split='train', **kwargs)
        self.val_dataset = LLFFDataset(split='val', **kwargs)  # val has different things with train

    def configure_optimizers(self):  # return two list (optimizer and scheduler)
        self.optimizer = get_optimizer(self.hparams, self.models)   # return a torch.optim
        scheduler = get_scheduler(self.hparams, self.optimizer)     # return a torch.optim.lr_scheduler
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time, and only have one image 
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):   # batch: output of dataloader, a tensor , tuple or list
        rays, rgbs = batch['rays'], batch['rgbs'] # shape (B, 8) , (B,3)
        results = self(rays)    # forward
        loss = self.loss(results, rgbs)         # return must have loss

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs = batch['rays'], batch['rgbs']
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays) # forward 
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)