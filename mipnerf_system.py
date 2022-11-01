import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

# model
from mip_nerf import MipNerf

# load dataset
from load_dataset_mip import LLFFDatasetmip, Rays_keys, Rays

# for torch and lightning
from lr_schedule import MipLRDecay

def calc_psnr(x: torch.Tensor, y: torch.Tensor):
    psnr = -10.0 * torch.log10(torch.mean((x - y) ** 2))
    return psnr

def stack_rgb(rgb_gt, coarse_rgb, fine_rgb):
    img_gt = rgb_gt.squeeze(0).permute(2, 0, 1).cpu()  # (3, H, W)
    coarse_rgb = coarse_rgb.squeeze(0).permute(2, 0, 1).cpu()
    fine_rgb = fine_rgb.squeeze(0).permute(2, 0, 1).cpu()

    stack = torch.stack([img_gt, coarse_rgb, fine_rgb])  # (3, 3, H, W)
    return stack

class MipNeRFSystem(LightningModule):   # render inside mipnerf, not mipnerfsystem
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.mip_nerf = MipNerf(
            resample_padding=hparams.nerf_resample_padding,
            disparity=hparams.nerf_disparity,
            min_deg_point=hparams.nerf_min_deg_point,
            max_deg_point=hparams.nerf_max_deg_point,
            deg_view=hparams.nerf_deg_view,
            density_noise=hparams.nerf_density_noise,
            density_bias=hparams.nerf_density_bias,
            rgb_padding=hparams.nerf_rgb_padding
        )

    def forward(self, batch_rays: torch.Tensor, randomized=True):
        res = self.mip_nerf(batch_rays, randomized)  # num_layers result
        return res

    def setup(self, stage):
        self.train_dataset = LLFFDatasetmip(data_dir=self.hparams.data_path, split='train')      
        self.val_dataset = LLFFDatasetmip(data_dir=self.hparams.data_path, split='val')

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.mip_nerf.parameters(), lr=self.hparams.lr,eps = 1e-8)
        scheduler = MipLRDecay(optimizer, self.hparams.lr, self.hparams.lr_final,
                               self.hparams.max_steps, self.hparams.optimizer_lr_delay_steps,
                               self.hparams.optimizer_lr_delay_mult)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.train_batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        # make the dataloader to an iter, so which can val n numbers images one time
        return iter(DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True))

    def training_step(self, batch, batch_nb):
        rays, rgbs_gt = batch # (B,?) (B,3)
        res = self(rays) # list[2],tuple[2], color (B,3), depth (B)

        # calculate loss for coarse and fine
        mask = rays.lossmult # (B,1) # i dont quite know what's this, it's all 1 anyway
        losses = []
        for (rgbs_pred, _) in res: # two tuples
            losses.append((mask * (rgbs_pred - rgbs_gt) ** 2).sum() / mask.sum())
        # The loss is a sum of coarse and fine MSEs
        mse_corse, mse_fine = losses
        loss = self.hparams.loss_coarse_loss_mult * mse_corse + mse_fine

        # calculate psnr for coarse and fine  # psnr no grad
        with torch.no_grad():
            psnrs = []
            for (rgbs_pred, _) in res:
                psnrs.append(calc_psnr(rgbs_pred, rgbs_gt))
            psnr_corse, psnr_fine = psnrs 

        self.log('lr', self.optimizers().optimizer.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_fine, prog_bar=True) # only take fine psnr

        return loss

    def validation_step(self, batch, batch_nb):
        _ , rgb_gt = batch  # notice that this is unflattened data
        coarse_rgb, fine_rgb, val_mask = self.render_image(batch)  # (1,H,W,3)

        val_mse_corse = (val_mask * (coarse_rgb - rgb_gt) ** 2).sum() / val_mask.sum()
        val_mse_fine = (val_mask * (fine_rgb - rgb_gt) ** 2).sum() / val_mask.sum()

        val_loss = self.hparams.loss_coarse_loss_mult * val_mse_corse + val_mse_fine
        val_psnr_corse = calc_psnr(coarse_rgb, rgb_gt)
        val_psnr_fine = calc_psnr(fine_rgb, rgb_gt)

        log = {'val/loss': val_loss, 'val/psnr': val_psnr_fine}
        stack = stack_rgb(rgb_gt, coarse_rgb, fine_rgb)  # (3,3,H,W)
        self.logger.experiment.add_images('val/GT_coarse_fine',
                                          stack, self.global_step)
        return log

    def validation_epoch_end(self, outputs):
        # not yet debugged..
        # validation is not quite crucial for nerf, i'll fix later
        # mean_loss = torch.stack([x['val/loss'] for x in outputs]).mean()
        # mean_psnr = torch.stack([x['val/psnr'] for x in outputs]).mean()
        mean_loss = 1
        mean_psnr = 1
        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)

    def render_image(self, batch):
        # batch is unflattened!
        rays, rgbs = batch
        _, H, W, _ = rgbs.shape  # (1,H,W,3)

        # change Rays to list: [origins, directions, viewdirs, radii, lossmult, near, far]
        single_image_rays = [getattr(rays, key) for key in Rays_keys]
        val_mask = single_image_rays[-3]

        # flatten each Rays attribute and put on device
        single_image_rays = [rays_attr.reshape(-1, rays_attr.shape[-1]) for rays_attr in single_image_rays]

        # get the amount of full rays of an image
        length = single_image_rays[0].shape[0]
        # divide each Rays attr into N groups according to chunk_size,
        # the length of the last group <= chunk_size
        single_image_rays = [[rays_attr[i:i + self.hparams.val_chunk_size] for i in range(0, length, self.hparams.val_chunk_size)] for
                             rays_attr in single_image_rays]
        # get N, the N for each Rays attr is the same
        length = len(single_image_rays[0])
        # generate N Rays instances
        single_image_rays = [Rays(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]

        corse_rgb, fine_rgb = [], []
        with torch.no_grad():
            for batch_rays in single_image_rays:
                (c_rgb, _), (f_rgb, _) = self(batch_rays, randomized=False)
                corse_rgb.append(c_rgb)
                fine_rgb.append(f_rgb)

        corse_rgb = torch.cat(corse_rgb, dim=0)
        fine_rgb = torch.cat(fine_rgb, dim=0)

        corse_rgb = corse_rgb.reshape(1, H, W, corse_rgb.shape[-1])  # 1 H W 3
        fine_rgb = fine_rgb.reshape(1, H, W, fine_rgb.shape[-1])  # 1 H W 3
        return corse_rgb, fine_rgb, val_mask
