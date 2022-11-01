import torch
import numpy as np
from einops import rearrange
from load_dataset_mip import Rays_keys, Rays

def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """
    Piecewise-Constant PDF sampling from sorted bins.
    Args:
        bins: torch.Tensor, [batch_size, num_bins + 1].
        weights: torch.Tensor, [batch_size, num_bins].
        num_samples: int, the number of samples.
        randomized: bool, use randomized samples.
    Returns:
        t_samples: torch.Tensor, [batch_size, num_samples+1].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)  # [B, 1]
    padding = torch.maximum(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.cumsum(pdf[..., :-1], dim=-1)
    cdf = torch.minimum(torch.ones_like(cdf), cdf)
    cdf = torch.cat([torch.zeros(list(cdf.shape[:-1]) + [1], device=cdf.device),
                     cdf,
                     torch.ones(list(cdf.shape[:-1]) + [1], device=cdf.device)],
                    dim=-1)  # [B, N]

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=cdf.device) * s)[None, ...]
        # u += jax.random.uniform(
        #     key,
        #     list(cdf.shape[:-1]) + [num_samples],
        #     maxval=s - jnp.finfo('float32').eps)
        u = u + torch.empty(list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(
            to=(s - torch.finfo(torch.float32).eps))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.full_like(u, 1. - torch.finfo(torch.float32).eps, device=u.device))
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, _ = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, _ = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = torch.clamp(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples

def lift_gaussian(directions, t_mean, t_var, r_var):
    """
        refer to eq.8-16 in paper
        Lift a Gaussian defined along a ray to 3D coordinates.
    Args:
        B: batch_size
        N: sample_num
        directions: (B,3)
        t_mean:     (B,N)
        t_var:      (B,N) 
        r_var:      (B,N)          
        DO NOT compute IPE cov because too expensive, directly go diagonal of IPE cov, and here, diagonal of cov
    """
    mean = directions[..., None, :] * t_mean[..., None]  # (B,1,3)*(B,N,1)=(B,N,3)

    # directly go diagonal, check eq.16
    d_outer_diag = directions ** 2  # (B,3) 
    d_norm_denominator = torch.sum(directions ** 2, dim=-1, keepdim=True) + 1e-10   # (B,1)
    null_outer_diag = 1 - d_outer_diag / d_norm_denominator # (B,3)
    t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]      # (B,N,1)*(B,1,3)=(B,N,3)
    xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]  # (B,N,1)*(B,1,3)=(B,N,3)
    cov_diag = t_cov_diag + xy_cov_diag

    return mean, cov_diag

def conical_frustum_to_gaussian(directions, t0, t1, base_radius):
    """
        refer to equations 7 and 8 in paper 
        why not having origins here?  --> no need, it's in cast_rays func        
    Args:
        B: batch_size
        N: sample_num
        directions: (B,3)
        t0:         (B,N) starting distance of the frustum.
        t1:         (B,N) ending distance of the frustum.
        base_radius:(B,1) r      
    Returns:
        a Gaussian (mean and covariance). without origin in eq.8
    """
    t_mu = (t0 + t1) / 2
    t_delta = (t1 - t0) / 2

    t_mean = t_mu + (2 * t_mu * t_delta ** 2) / (3 * t_mu ** 2 + t_delta ** 2)
    t_var = (t_delta ** 2) / 3 - (4 / 15) * ((t_delta ** 4 * (12 * t_mu ** 2 - t_delta ** 2)) /
                                        (3 * t_mu ** 2 + t_delta ** 2) ** 2)
    r_var = base_radius ** 2 * ((t_mu ** 2) / 4 + (5 / 12) * t_delta ** 2 - 4 / 15 *
                                (t_delta ** 4) / (3 * t_mu ** 2 + t_delta ** 2))

    return lift_gaussian(directions, t_mean, t_var, r_var)

def cast_rays(t_samples, origins, directions, radii):
    """
        Cast cone rays and featurize sections of it.
    Args:
        B: batch_size
        N: sample_num
        t_samples:  (B,N+1) the "fencepost" distances along the ray.
        origins:    (B,3)
        directions: (B,3)
        radii:      (B,1)
    Returns:
        a tuple of arrays of means and covariances.
    """
    t0 = t_samples[..., :-1]  # (B,N)
    t1 = t_samples[..., 1:]   # (B,N) 

    means, covs = conical_frustum_to_gaussian(directions, t0, t1, radii) # (B,N,3)
    means = means + origins[..., None, :] # (B,N,3)+(B,1,3)=(B,N,3)
    return means, covs

def sample_along_rays(origins, directions, radii, near, far, randomized, disparity, num_samples = 128):
    """
    Stratified sampling along the rays.
    Args:
        B: batch_size
        N: sample_num
        origins:    o (B,3) Tensor
        directions: d (B,3) Tensor
        radii:      r (B,1) Tensor
        near:       n (B,1) Tensor, would be 0 in NDC
        far:        f (B,1) Tensor, would be 1 in NDC
        randomized: randomized stratified sampling.
        disparity:  sampling linearly in disparity rather than depth.
        num_samples:128
    Returns:
        t_samples:  (B,N+1) coarse sample
        means:      (B,N,3)
        covs:       (B,N,3)
    """
    batch_size = origins.shape[0]

    t_samples = torch.linspace(0., 1., num_samples + 1, device=origins.device)
    # plus one because it's not dot, but region # check the paper
    if disparity:  # this is not working actually for NDC now, try not to use
        t_samples = 1. / ((1. - t_samples) / (near+0.001) + t_samples / far)
    else:
        t_samples = near + (far - near) * t_samples

    if randomized:
        mids = 0.5 * (t_samples[..., 1:] + t_samples[..., :-1])
        upper = torch.cat([mids, t_samples[..., -1:]], -1)  # (1,129)
        lower = torch.cat([t_samples[..., :1], mids], -1)   # (1,129)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device) # (B,129)
        t_samples = lower + (upper - lower) * t_rand    # (B,129)
    else:
        # Broadcast t_samples to make the returned shape consistent.
        t_samples = torch.broadcast_to(t_samples, [batch_size, num_samples + 1]) # (B,129)
    means, covs = cast_rays(t_samples, origins, directions, radii)  # (B,N,3)
    return t_samples, (means, covs)

def resample_along_rays(origins, directions, radii, t_samples, weights, randomized, resample_padding):
    """Resampling.
    Args:
        B: batch_size
        N: sample_num
        origins:    o (B,3) Tensor
        directions: d (B,3) Tensor
        radii:      r (B,1) Tensor
        t_samples:    (B,N+1) from coarse
        weights:      (B,N) from coarse for t_samples
        randomized:   use randomized samples.
        resample_padding:  added to the weights before normalizing.
    Returns:
        t_samples:    (B,N+1) fine one
        means:        (B,N,3)
        covs:         (B,N,3)
    """

    # stop grad, not to backprop through sampling
    with torch.no_grad():
        # eq.18
        weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:]) + resample_padding

        new_t_vals = sorted_piecewise_constant_pdf( t_samples,
                                                    weights,
                                                    t_samples.shape[-1],
                                                    randomized) # (B,N+1)

    means, covs = cast_rays(new_t_vals, origins, directions, radii)  # (B,N,3)
    return new_t_vals, (means, covs)

# above eq.8, below eq.14

def integrated_pos_enc(means_covs, min_deg, max_deg):
    """Encode `means` with sinusoids scaled by 2^[min_deg:max_deg-1].
    Args:
        means_covs: a tuple 
            means:          (B,N,3)  , variables to be encoded. Should be in [-pi, pi]. 
            covs_diagonal:  (B,N,3), default has diagonal
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
    Returns:
        encoded: torch.Tensor, encoded variables.
    """

    means, covs_diag = means_covs # (B,N,3)

    # eq.9 - P (L)
    P = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=means.device) # (L)

    # eq.10
    #（B,N,1,3)*(L,1) = (B,N,L,3)
    PElift_mean = means[..., None, :] * P[:, None]
    # (B,N,L,3)->(B,N,3L)
    PElift_mean = rearrange(PElift_mean, 'B N L mean_dim -> B N (L mean_dim)')  # (B,N,3L)

    # eq.15
    #（B,N,1,3)*(L,1) = (B,N,L,3)
    PElift_cov_diagonal = covs_diag[..., None, :] * P[:, None] ** 2
    # (B,N,L,3)->(B,N,3L)
    PElift_cov_diagonal = rearrange(PElift_cov_diagonal,'B N L cov_dim -> B N (L cov_dim)') # (B,N,3L)

    # eq.14
    # sin(y + 0.5 * torch.tensor(np.pi)) = cos(y)
    sin_cos_PElift_mean = torch.cat([PElift_mean, PElift_mean + 0.5 * torch.tensor(np.pi)], dim=-1) # (B,N,6L)
    sin_cos_PElift_cov_diagonal = torch.cat([PElift_cov_diagonal] * 2, dim=-1) # (B,N,6L)
    IPE = torch.exp(-0.5 * sin_cos_PElift_cov_diagonal) * torch.sin(sin_cos_PElift_mean) # (B,N,6L)

    return IPE

def pos_enc(dirs, min_deg, max_deg):
    """
        The positional encoding used by the original NeRF paper.
        dirs: (B,3)
        default add identity
    """
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=dirs.device)
    #（B,1,3)*(L,1) = (B,L,3)
    lift_dir = dirs[..., None, :] * scales[:, None]
    # (B,L,3) -> (B,3L)
    lift_dir = rearrange(lift_dir,'B L x_dim -> B (L x_dim)') # (B,3L)
    sin_lift_dir = torch.sin(torch.cat([lift_dir, lift_dir + 0.5 * torch.tensor(np.pi)], dim=-1))  # (B,6L)

    return torch.cat([dirs] + [sin_lift_dir], dim=-1)  # (B,6L+3)

# below volume rendering

def volumetric_rendering(rgb, density, t_samples, dirs):
    """
        Volumetric Rendering Function.
    Args:
        B: batch
        N: #samples
        rgb:      (B,N,3)
        density:  (B,N,1)
        t_samples:(B,N+1)
        dirs:     (B,3)
    Returns:
        final_rgb:(B,3)
        distance: (B)
        weights:  (B,N)
    """


    t_mids = 0.5 * (t_samples[..., :-1] + t_samples[..., 1:])   # (B,N)
    t_interval = t_samples[..., 1:] - t_samples[..., :-1]       # (B,N)

    # normalize dirs (again ?)
    dirs = torch.linalg.norm(dirs[..., None, :], dim=-1) # (B,1)

    delta = t_interval * dirs # (B,N)
    density = density[..., 0] # (B,N)
    density_delta = density * delta # (B,N)

    alphas = 1 - torch.exp(-density_delta) # (B,N)
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1)

    # output weights
    weights = alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (B,N)
    # output rgb
    final_rgb = (weights[..., None]*rgb).sum(axis=-2)  # (B,N,1)*(B,N,3) -> (B,3)
    # output dist
    distance = torch.nan_to_num((weights * t_mids).sum(axis=-1) / weights.sum(axis=-1)) # (B)
    # distance = torch.clamp(distance, t_samples[:, 0], t_samples[:, -1]) is not working
    distance = torch.where(distance<t_samples[:, 0],t_samples[:, 0],distance)
    distance = torch.where(distance>t_samples[:, -1],t_samples[:, -1],distance)

    return final_rgb, distance, weights

# add for render final

def rearrange_render_image(rays, chunk_size=4096):
    # change Rays to list: [origins, directions, viewdirs, radii, lossmult, near, far]
    single_image_rays = [getattr(rays, key) for key in Rays_keys]
    val_mask = single_image_rays[-3]

    # flatten each Rays attribute and put on device
    single_image_rays = [rays_attr.reshape(-1, rays_attr.shape[-1]) for rays_attr in single_image_rays]
    # get the amount of full rays of an image
    length = single_image_rays[0].shape[0]
    # divide each Rays attr into N groups according to chunk_size,
    # the length of the last group <= chunk_size
    single_image_rays = [[rays_attr[i:i + chunk_size] for i in range(0, length, chunk_size)] for
                         rays_attr in single_image_rays]
    # get N, the N for each Rays attr is the same
    length = len(single_image_rays[0])
    # generate N Rays instances
    single_image_rays = [Rays(*[rays_attr[i] for rays_attr in single_image_rays]) for i in range(length)]
    return single_image_rays, val_mask