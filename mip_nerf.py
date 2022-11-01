import torch
from torch import nn
from einops import repeat
from collections import namedtuple

# real mip
from mip import sample_along_rays, resample_along_rays, integrated_pos_enc, pos_enc, volumetric_rendering

def xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    nn.init.xavier_uniform_(linear.weight.data)


class MLP(nn.Module):      # network has pretty much the same shape as nerf, please check nerf supple material for a reference
    def __init__(   self,
                    xyz_dim: int,
                    dir_dim: int,
                    net_depth: int=8,               # The depth of the first part of MLP.
                    net_width: int=256,             # The width of the first part of MLP.
                    net_depth_condition: int=1,     # The depth of the second part of MLP.
                    net_width_condition: int=128,   # The width of the second part of MLP.
                    skip_index: int=4,              # a skip connection default 4
                    loading_model = False):
        super(MLP, self).__init__()
        self.skip_index = skip_index if not loading_model else skip_index+1

        #first part MLP 8 layers
        layers = []
        for i in range(net_depth):
            if i == 0:  # first layer
                dim_in = xyz_dim
                dim_out = net_width
            elif i == self.skip_index:   # this is simplified.. only consider one skip layer at exact
                dim_in = net_width + xyz_dim
                dim_out = net_width
            else:
                dim_in = net_width
                dim_out = net_width

            linear = nn.Linear(dim_in, dim_out)
            xavier_init(linear)
            layers.append(nn.Sequential(linear, nn.ReLU(True)))

        self.layers = nn.ModuleList(layers)

        # density output
        self.density_layer = nn.Linear(net_width, 1) # density is 1 channel 
        xavier_init(self.density_layer)

        #second part MLP 1 layer output color
        linear = nn.Linear(net_width + dir_dim, net_width_condition)
        xavier_init(linear)
        self.view_layers = nn.Sequential(linear, nn.ReLU(True))

        # color output
        self.color_layer = nn.Linear(net_width_condition, 3) #color is 3 channels
        xavier_init(self.color_layer)

    def forward(self, x, view_direction):
        """Evaluate the MLP.
        Args:
            B: batch
            N: #samples
            x:              (B,N,6L)
            view_direction: (B,6L+3)
        Returns:
            rgb:     (B,N,3)
            density: (B,N,1)
        """
        N = x.shape[1]
        inputs = x  # (B,N,6L)

        # through first 8 MLP
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i+1 == self.skip_index: # careful i+1 .. next layer taking skipping input
                x = torch.cat([x, inputs], dim=-1)

        # output density
        density = self.density_layer(x)  # (B,N,1)

        # x (B,N,6L), lift view_direction to (B,N,...)
        view_direction = repeat(view_direction, 'B feature -> B sample feature', sample=N)
        x = torch.cat([x, view_direction], dim=-1)
        x = self.view_layers(x)
        rgb = self.color_layer(x)

        return rgb, density


class MipNerf(nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""

    def __init__(self, 
                 resample_padding: float = 0.01,
                 disparity: bool = False,
                 min_deg_point: int = 0,
                 max_deg_point: int = 16,
                 deg_view: int = 4,
                 density_noise: float = 0.,
                 density_bias: float = -1.,
                 rgb_padding: float = 0.001,
                 loading_model = False):
        super(MipNerf, self).__init__()
        self.disparity = disparity              # If True, sample linearly in disparity, not in depth.
        self.min_deg_point = min_deg_point      # Min degree of positional encoding for 3D points.
        self.max_deg_point = max_deg_point      # Max degree of positional encoding for 3D points.
        self.deg_view = deg_view                # Degree of positional encoding for viewdirs.
        self.density_noise = density_noise      # Standard deviation of noise added to raw density.
        self.density_bias = density_bias        # The shift added to raw densities pre-activation.
        self.resample_padding = resample_padding# alpha "padding" on the histogram.
        self.rgb_padding = rgb_padding          # ?
        # please ignore the loading_model param and dn not tune

        mlp_xyz_dim = (max_deg_point - min_deg_point) * 3 * 2   # not appending identity
        mlp_dir_dim = (deg_view * 2 + 1) * 3                    # append identity already

        self.mlp = MLP(xyz_dim = mlp_xyz_dim, 
                       dir_dim = mlp_dir_dim,
                       loading_model = loading_model)
        
        self.rgb_activation = torch.nn.Sigmoid()
        self.density_activation = torch.nn.Softplus()


    def forward(self, rays: namedtuple, randomized: bool):
        """The mip-NeRF Model.
        Args:
            rays is a namedtuple with:
                'origins' : o
                'directions': d
                'viewdirs': d normalized
                'radii': r
                'near': NDC it's all 0 
                'far': NDC it's all 1
            randomized: bool, use randomized stratified sampling.
        Returns:
            res: list, [*(rgb, distance, acc)]
        Logic notes:
            the mipnerf system will compute loss and psnr
            WITH the querying result from this mipnerf model
            this model only outputs color mostly
        """

        res = []
        t_samples, weights = None, None     # keep it

        for i_level in range(2):  # coarse and fine, in two
            if i_level == 0: # coarse
                # Stratified sampling along rays
                t_samples, means_covs = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    rays.near,
                    rays.far,
                    randomized, 
                    self.disparity
                )
            else: # fine
                t_samples, means_covs = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_samples,
                    weights,
                    randomized,
                    resample_padding=self.resample_padding
                )

            # t_samples (B,N+1)
            # means_covs tuple, two (B,N,3)

            # actually getting mean and diag_cov here, next step compute eq.13/14,15

            # IPE, getting (B,N,2*3*L) L:(max_deg_point - min_deg_point)
            samples_enc = integrated_pos_enc(means_covs,
                                            self.min_deg_point,
                                            self.max_deg_point)

            # Point attribute predictions, using viewdirs
            viewdirs_enc = pos_enc(rays.viewdirs,
                                    min_deg=0,
                                    max_deg=self.deg_view)

            # samples_enc (B,N,6L)
            # viewdirs_enc (B,6L+3)
            raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)
            # raw_rgb:     (B,N,3)
            # raw_density: (B,N,1)

            # Add noise to regularize the density predictions if needed.
            if randomized and (self.density_noise > 0):
                raw_density += self.density_noise * torch.randn(raw_density.shape, dtype=raw_density.dtype)

            # Volumetric rendering.
            rgb = self.rgb_activation(raw_rgb) # (B,N,3)
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding # well
            density = self.density_activation(raw_density + self.density_bias)  # (B,N,1)
            final_rgb, distance, weights = volumetric_rendering( rgb,
                                                                density,
                                                                t_samples,
                                                                rays.directions)
            # weights for resample, output else
            res.append((final_rgb, distance))  # list[2] each with tuple(2)

        return res