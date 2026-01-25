import json
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.config import args as ModelArgs
from utils.Traj_UNet import *
from utils.utils import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--test_trajectories_path", type=Path, required=True)
    parser.add_argument("--test_attributes_path", type=Path, required=True)
    parser.add_argument("--statistics_path", type=Path, required=True)
    parser.add_argument("--num_test_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def load_statistics(statistics_path):
    with open(statistics_path, "r") as f:
        stats = json.load(f)

    head_mean = [
        0,
        stats["trip_distance"]["mean"],
        stats["trip_time"]["mean"],
        stats["trip_length"]["mean"],
        stats["avg_dis"]["mean"],
        stats["avg_speed"]["mean"],
    ]
    head_std = [
        1,
        stats["trip_distance"]["std"],
        stats["trip_time"]["std"],
        stats["trip_length"]["std"],
        stats["avg_dis"]["std"],
        stats["avg_speed"]["std"],
    ]
    geo_mean = np.array([stats["lon"]["mean"], stats["lat"]["mean"]])
    geo_std = np.array([stats["lon"]["std"], stats["lat"]["std"]])
    len_mean = stats["traj_len"]["mean"]
    len_std = stats["traj_len"]["std"]

    return (head_mean, head_std, geo_mean, geo_std, len_mean, len_std)


def main(args):
    temp = {k: SimpleNamespace(**v) for k, v in ModelArgs.items()}
    config = SimpleNamespace(**temp)

    unet = Guide_UNet(config).cuda()
    unet.load_state_dict(torch.load(args.model_path))

    # sampling parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, n_steps).cuda()
    eta = 0.0
    timesteps = 100
    skip = n_steps // timesteps
    seq = range(0, n_steps, skip)

    # load test trajectories
    traj = np.load(args.test_trajectories_path, allow_pickle=True)
    traj = traj[:, :, :2]
    traj = np.swapaxes(traj, 1, 2)  # [N, 2, 200]
    traj = torch.from_numpy(traj).float()

    # load head information for guide trajectory generation
    head = np.load(args.test_attributes_path, allow_pickle=True)
    head = torch.from_numpy(head).float()  # [N, 8]

    # get test samples
    traj = traj[: args.num_test_samples, :, :]
    head = head[: args.num_test_samples, :]
    dataset = TensorDataset(traj, head)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # load statistics
    _, _, geo_mean, geo_std, len_mean, len_std = load_statistics(args.statistics_path)

    all_label_traj, all_gen_trajs = [], []
    for batch_traj, batch_head in tqdm(dataloader):
        lengths = batch_head[:, 3]
        lengths = lengths * len_std + len_mean
        lengths = lengths.int()
        batch_head = batch_head.cuda()

        # Start with random noise
        x = torch.randn(args.batch_size, 2, config.data.traj_length).cuda()
        ims = []
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            with torch.no_grad():
                pred_noise = unet(x, t, batch_head)
                x = p_xt(x, pred_noise, t, next_t, beta, eta)
                if i % 10 == 0:
                    ims.append(x.cpu().squeeze(0))

        gen_trajs = ims[-1].cpu().numpy()
        gen_trajs = gen_trajs[:, :2, :]  # [B, 2, 200]

        # resample the trajectory length
        for j in range(args.batch_size):
            resampled_label_traj = resample_trajectory(batch_traj[j].T, lengths[j])
            resampled_label_traj = resampled_label_traj * geo_std + geo_mean
            all_label_traj.append(resampled_label_traj)

            resampled_gen_trajs = resample_trajectory(gen_trajs[j].T, lengths[j])
            resampled_gen_trajs = resampled_gen_trajs * geo_std + geo_mean
            all_gen_trajs.append(resampled_gen_trajs)

    # save generated trajectories
    results_path = args.model_path.parent.parent.parent / "results"
    np.save(results_path / "label_trajs.npy", np.array(all_label_traj, dtype=object), allow_pickle=True)
    np.save(results_path / "gen_trajs.npy", np.array(all_gen_trajs, dtype=object), allow_pickle=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
