import torch
import numpy as np

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = torch.sigmoid(torch.from_numpy(betas)).numpy() * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Diffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        device="cpu"
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.device = device
        
        betas = get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps)
        self.betas = torch.from_numpy(betas).float().to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.max(self.posterior_variance, torch.tensor(1e-20, device=device))
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_diffusion_timesteps, (batch_size,), device=self.device)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, y, cfg_scale=1.0):
        """
        Sample from the model (reverse process)
        """
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1, 1)
        
        # Model output
        # Handle Classifier-Free Guidance
        if cfg_scale > 1.0:
            # Create null class label (usually num_classes)
            # Assuming y has shape (N,)
            y_null = torch.ones_like(y) * 1000 # 1000 is null class if num_classes=1000
            
            # Double input for batched inference
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            y_in = torch.cat([y, y_null], dim=0)
            
            model_output = model(x_in, t_in, y_in)
            
            # DiT predicts both noise and covariance if learn_sigma=True
            # Output shape: (2*N, 2*C, H, W) or (2*N, C, H, W)
            
            eps, rest = model_output.chunk(2, dim=0)
            # Further chunk if learning variance
            if model.out_channels > model.in_channels:
                 # Standard DiT: predicts [noise, covariance]
                 # We only care about noise for simple sampling usually, 
                 # but for improved sampling we use the covariance.
                 # Let's simplify: just use noise prediction part for mean
                 eps, _ = eps.chunk(2, dim=1)
                 eps_null, _ = rest.chunk(2, dim=1)
            else:
                 eps_null = rest
            
            eps = eps_null + cfg_scale * (eps - eps_null)
        else:
            model_output = model(x, t, y)
            if model.out_channels > model.in_channels:
                eps, _ = model_output.chunk(2, dim=1)
            else:
                eps = model_output

        # Calculate mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size, y, channels=4, cfg_scale=1.0):
        device = self.device
        img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        
        for i in reversed(range(0, self.num_diffusion_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i, y, cfg_scale=cfg_scale)
            
        return img
