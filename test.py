import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion,DiffusiveRestoration
from tqdm import tqdm
import torch.nn as nn
def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Evaluate Wavelet-Based Diffusion Model')
    parser.add_argument("--config", default='/home/ubuntu/Low/configs/low-light.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='/home/ubuntu/Low/real/LOL-v2-Real.pth.tar', type=str,
                        help='Path for the checkpoint to load for evaluation')
    parser.add_argument("--sampling_timesteps", type=int, default=10,
                        help="Number of implicit sampling steps")
    parser.add_argument("--image_folder", default='', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=20826, type=int, metavar='N',
                        help='Seed for initializing training')
    parser.add_argument('--mode', default='test', type=str, metavar='N')
    parser.add_argument('--use_kind_align', default='false', type=str, metavar='N')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Current device: {}".format(device))
    if torch.cuda.is_available():
       current_gpu = torch.cuda.current_device()
       gpu_name = torch.cuda.get_device_name(current_gpu)
       print("Current GPU: {} - {}".format(current_gpu, gpu_name))
    config.device = device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("Current dataset '{}'".format(config.data.val_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()
    
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    # model = nn.DataParallel(model)
    val_loader_with_progress = tqdm(val_loader, desc='Loading Validation Data', leave=False)
    model.restore(val_loader_with_progress)

if __name__ == '__main__':
    main()
