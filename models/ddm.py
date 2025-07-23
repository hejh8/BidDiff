import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ssim
import utils
from models.unet import DualDiffusionUNet
from models.Rcolor import RetinexColorCorrectionNet



class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

    def state_dict(self):
        return self.shadow





class Net(nn.Module):
    def __init__(self, args, config):
        super(Net, self).__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.Unet = DualDiffusionUNet(config).to(self.device)
        betas = utils.scheduler.get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = self.betas.shape[0]

        self.RCC = RetinexColorCorrectionNet()
        self.test = True

    @staticmethod
    def compute_alpha(beta, t):
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, dm_num=True, eta=0.):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        seq_next = [-1] + list(seq[:-1])

        n, c, h, w = x_cond.shape
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((n,), i, device=x.device)
            next_t = torch.full((n,), j, device=x.device)
            at = self.compute_alpha(b, t.long())
            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1]

            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)

        return xs

    def forward(self, x):
        data_dict = {}
        input_img = x[:, :3, :, :]
        input_img_norm = utils.sampling.data_transform(input_img)
        b = self.betas.to(self.device)

        if self.test:
            img_list = self.sample_training(input_img_norm, b)
            pred_x_g = img_list[-1]
            pred_x_g = utils.sampling.inverse_data_transform(pred_x_g)
            pred_x_g = self.RCC(pred_x_g)
            data_dict["pred_x"] = pred_x_g

        return data_dict


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device
        self.model = Net(args, config).to(self.device)
        self.model = torch.nn.DataParallel(self.model)
        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("Loaded checkpoint:", load_path)

    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.type + str(self.config.data.patch_size))
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                b, _, img_h, img_w = x.shape
                img_h_32 = int(32 * np.ceil(img_h / 32.0))
                img_w_32 = int(32 * np.ceil(img_w / 32.0))
                x = F.pad(x, (0, img_w_32 - img_w, 0, img_h_32 - img_h), 'reflect')

                out = self.model(x.to(self.device))
                pred_x = out["pred_x"]
                pred_x = pred_x[:, :, :img_h, :img_w]

                utils.logging.save_image(pred_x, os.path.join(image_folder, str(step), str(1), f"{y[0]}.png"))


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader):
        image_folder = os.path.join(self.args.image_folder, self.config.data.val_dataset)
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                b, c, h, w = x_cond.shape
                img_h_32 = int(32 * np.ceil(h / 32.0))
                img_w_32 = int(32 * np.ceil(w / 32.0))
                x_cond = F.pad(x_cond, (0, img_w_32 - w, 0, img_h_32 - h), 'reflect')

                x_output = self.diffusion.model(x_cond)["pred_x"]
                x_output = x_output[:, :, :h, :w]
                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y[0]}.png"))
