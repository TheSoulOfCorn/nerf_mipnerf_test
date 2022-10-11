import torch
from einops import rearrange, reduce, repeat

__all__ = ['render_rays']

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2" # all mid
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    # print('pdf:',pdf)
    # print('cdf:',cdf)
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    # print(u)
    inds = torch.searchsorted(cdf, u, right=True) # (N_ray, N_imp)    # less than this one cdf
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)
    # print('inds:',inds)
    # print('below:',below)
    # print('above:',above)
    # cdf : Nsampe +1 range from 0-1

    # cdf : (N_rays, Nsampe +1)  # inds_sampled (N_rays,N_imp *2)
    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    # print('inds_sampled:',inds_sampled)
    # print('cdf_g:',cdf_g)
    # print('bins_g:',bins_g)   
    denom = cdf_g[...,1]-cdf_g[...,0]
    # print('denom:',denom)   
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0,
                         # in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])  # ?
    return samples

def render_rays(models,
                embeddings,
                rays,           # chunked shape: (C * 8) :  o, d, near, far
                N_samples=64,   # coarse
                use_disp=False, # 
                perturb=1,      # default 1 
                noise_std=1,
                N_importance=0, # 128
                chunk=1024*32,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    #inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals, test_time, **kwargs)
    def inference(results, model, typ, xyz, z_vals,**kwargs):
        """
        real volume render step
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
                        # (N_rays*N_samples_, embed_dir_channels)
        for i in range(0, B, chunk):
            xyz_embedded = embedding_xyz(xyz_[i:i+chunk])
            xyzdir_embedded = torch.cat([xyz_embedded,
                                            dir_embedded_[i:i+chunk]], 1)
            out_chunks += [model(xyzdir_embedded)]

        out = torch.cat(out_chunks, 0)
        # out = out.view(N_rays, N_samples_, 4)
        out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_, c=4) # just for clarity
        rgbs = out[..., :3] # (N_rays, N_samples_, 3)
        sigmas = out[..., 3] # (N_rays, N_samples_)
        

        # Convert these values using volume rendering (Section 4)
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha by the formula (3)
        noise = torch.randn_like(sigmas) * noise_std
        alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)
        # get T
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, 1-a1, 1-a2, ...]
        # each one has shape as example: (1-a1)(1-a2)a3
        weights = \
            alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum') # (N_rays), the accumulated opacity along the rays
                                                            # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        results[f'z_vals_{typ}'] = z_vals

        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*rgbs, 'n1 n2 c -> n1 c', 'sum')
        depth_map = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')

        results[f'rgb_{typ}'] = rgb_map
        results[f'depth_{typ}'] = depth_map

        return

    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
    # Embed direction
    #dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d)) # (N_rays, embed_dir_channels)
    dir_embedded = embedding_dir(rays_d)

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)
    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
    
    if perturb > 0: # perturb sampling depths (z_vals)  # just get random!
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand
    # rays_d n1 1 c (N 1 3)    z_vals  (N n_sample 1)
    # * : get (N n_sample 3)  every o,d has a (n_sample,3) xyz loc
    xyz_coarse = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

    results = {}
    inference(results, models['coarse'], 'coarse', xyz_coarse, z_vals,**kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here
        z_vals = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)[0]
                 # combine coarse and fine samples
        xyz_fine = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        inference(results, models['fine'], 'fine', xyz_fine, z_vals,**kwargs)

    return results
