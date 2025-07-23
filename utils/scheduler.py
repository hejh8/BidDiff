import numpy as np

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x): return 1 / (np.exp(-x) + 1)
    
    if beta_schedule == "quad":
        return np.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps)**2
    elif beta_schedule == "linear":
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps)
    elif beta_schedule == "const":
        return beta_end * np.ones(num_diffusion_timesteps)
    elif beta_schedule == "jsd":
        return 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps)
    elif beta_schedule == "sigmoid":
        return sigmoid(np.linspace(-6, 6, num_diffusion_timesteps)) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"{beta_schedule} is not supported.")
