# This code part from https://github.com/sunlin-ai/diffusion_tutorial

import datetime
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.config import args as model_args
from utils.EMA import EMAHelper
from utils.logger import Logger, log_info
from utils.Traj_UNet import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("data/"))
    parser.add_argument("--city", type=str, choices=["Beijing", "Porto", "San_Francisco"])
    parser.add_argument("--trajectory_length", type=int, default=200)
    return parser.parse_args()


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def main(config, logger, exp_dir):
    # Modified to return the noise itself as well
    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps  # also returns noise

    # Create the model
    unet = Guide_UNet(config).cuda()
    unet = torch.compile(unet)

    ###########################################################
    # The input shape of traj and head list as follows:
    # traj: [batch_size, 2, traj_length]   2: latitude and longitude
    # head: [batch_size, 8]   8: departure_time, trip_distance,  trip_time, trip_length, avg_dis, avg_speed, start_id, end_id
    ###########################################################
    train_traj = np.load(config.data.train_traj_path, allow_pickle=True)
    train_head = np.load(config.data.train_head_path, allow_pickle=True)
    train_traj = np.swapaxes(train_traj, 1, 2)
    train_traj = torch.from_numpy(train_traj).float()
    train_head = torch.from_numpy(train_head).float()

    train_dataset = TensorDataset(train_traj, train_head)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    eval_traj = np.load(config.data.eval_traj_path, allow_pickle=True)
    eval_head = np.load(config.data.eval_head_path, allow_pickle=True)
    eval_traj = np.swapaxes(eval_traj, 1, 2)
    eval_traj = torch.from_numpy(eval_traj).float()
    eval_head = torch.from_numpy(eval_head).float()
    eval_dataset = TensorDataset(eval_traj, eval_head)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, n_steps).cuda()
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    lr = 2e-4  # Explore this - might want it lower when training on the full dataset
    best_val_loss = float("inf")
    min_delta = 1e-6
    patience = 5
    stop_counter = 0

    # optimizer
    optim = torch.optim.AdamW(unet.parameters(), lr=lr)  # Optimizer

    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None

    # new filefold for save model pt
    model_save = exp_dir / "models" / (timestamp + "/")
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    scaler = torch.amp.GradScaler()

    # config.training.n_epochs = 1
    pbar = tqdm(total=config.training.n_epochs * len(train_dataloader))
    for epoch in range(1, config.training.n_epochs + 1):
        unet.train()
        pbar.set_description(f"Epoch {epoch}/{config.training.n_epochs}")
        for _, (trainx, head) in enumerate(train_dataloader):
            x0 = trainx.cuda()
            head = head.cuda()
            t = torch.randint(low=0, high=n_steps, size=(len(x0) // 2 + 1,)).cuda()
            t = torch.cat([t, n_steps - t - 1], dim=0)[: len(x0)]
            # Get the noised images (xt) and the noise (our target)
            xt, noise = q_xt_x0(x0, t)

            optim.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Run xt through the network to get its predictions
                pred_noise = unet(xt.float(), t, head)
                # Compare the predictions with the targets
                loss = F.mse_loss(noise.float(), pred_noise)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if config.model.ema:
                ema_helper.update(unet)

            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)

        if epoch % 10 == 0:
            unet.eval()
            val_loss_list = []
            val_pbar = tqdm(eval_dataloader, desc=f"Validating Epoch {epoch}")
            with torch.no_grad():
                for val_x, val_head in val_pbar:
                    val_x = val_x.cuda()
                    val_head = val_head.cuda()
                    t_val = torch.randint(0, n_steps, (val_x.shape[0],), device=val_x.device)
                    xt_val, noise_val = q_xt_x0(val_x, t_val)

                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred_val = unet(xt_val.float(), t_val, val_head)
                        v_loss = F.mse_loss(noise_val.float(), pred_val)
                    val_loss_list.append(v_loss.item())
                    val_pbar.set_postfix({"val_loss": v_loss.item()})

            avg_val_loss = np.mean(val_loss_list)
            if avg_val_loss < (best_val_loss - min_delta):
                best_val_loss = avg_val_loss
                stop_counter = 0
                torch.save(unet.state_dict(), model_save / "unet_best.pt")
            else:
                stop_counter += 1

            torch.save(unet.state_dict(), model_save / f"unet_{epoch}.pt")
            if stop_counter >= patience:
                print(f"Early stopping at epoch {epoch} with best validation loss: {best_val_loss:.6f}")
                break


if __name__ == "__main__":
    args = parse_args()
    config = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in model_args.items()})
    config.data.dataset = args.city
    config.data.train_traj_path = args.input_dir / args.city / "train_trajectories.npy"
    config.data.train_head_path = args.input_dir / args.city / "train_attrs.npy"
    config.data.eval_traj_path = args.input_dir / args.city / "val_trajectories.npy"
    config.data.eval_head_path = args.input_dir / args.city / "val_attrs.npy"
    config.data.traj_length = args.trajectory_length

    root_dir = Path(__name__).resolve().parents[0]
    result_name = "{}_steps={}_len={}_{}_bs={}".format(
        config.data.dataset,
        config.diffusion.num_diffusion_timesteps,
        config.data.traj_length,
        config.diffusion.beta_end,
        config.training.batch_size,
    )
    exp_dir = root_dir / "models" / result_name
    for d in ["results", "models", "logs", "Files"]:
        os.makedirs(exp_dir / d, exist_ok=True)
    print("All files saved path ---->>", exp_dir)
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    files_save = exp_dir / "Files" / (timestamp + "/")
    if not os.path.exists(files_save):
        os.makedirs(files_save)
    shutil.copy("./utils/config.py", files_save)
    shutil.copy("./utils/Traj_UNet.py", files_save)

    logger = Logger(__name__, log_path=exp_dir / "logs" / (timestamp + ".log"), colorize=True)
    log_info(config, logger)
    main(config, logger, exp_dir)
