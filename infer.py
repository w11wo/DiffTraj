import json
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.config import args as model_args
from utils.Traj_UNet import *
from utils.utils import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--input_dir", type=Path, default=Path("data/"))
    parser.add_argument("--city", type=str, choices=["Beijing", "Porto", "San_Francisco"])
    parser.add_argument("--trajectory_length", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def load_statistics(statistics_path):
    with open(statistics_path, "r") as f:
        stats = json.load(f)

    head_mean = [
        0,
        stats["trip_distance_m"]["mean"],
        stats["trip_time_min"]["mean"],
        stats["trip_length"]["mean"],
        stats["avg_dis_m"]["mean"],
        stats["avg_speed_m_s"]["mean"],
    ]
    head_std = [
        1,
        stats["trip_distance_m"]["std"],
        stats["trip_time_min"]["std"],
        stats["trip_length"]["std"],
        stats["avg_dis_m"]["std"],
        stats["avg_speed_m_s"]["std"],
    ]
    geo_mean = np.array([stats["lon"]["mean"], stats["lat"]["mean"]])
    geo_std = np.array([stats["lon"]["std"], stats["lat"]["std"]])
    len_mean = stats["traj_len"]["mean"]
    len_std = stats["traj_len"]["std"]

    return (head_mean, head_std, geo_mean, geo_std, len_mean, len_std)


def main(args):
    config = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in model_args.items()})
    config.data.dataset = args.city
    config.data.eval_traj_path = args.input_dir / args.city / "test_trajectories.npy"
    config.data.eval_head_path = args.input_dir / args.city / "test_attrs.npy"
    config.data.traj_length = args.trajectory_length

    unet = Guide_UNet(config).cuda().eval()
    unet = torch.compile(unet)
    unet.load_state_dict(torch.load(args.model_path))

    # sampling parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, n_steps).cuda()
    eta = 0.0
    timesteps = 100
    skip = n_steps // timesteps
    seq = range(0, n_steps, skip)

    # load test trajectories
    traj = np.load(config.data.eval_traj_path, allow_pickle=True)
    traj = traj[:, :, :2]
    traj = np.swapaxes(traj, 1, 2)  # [N, 2, 200]
    traj = torch.from_numpy(traj).float()

    # load head information for guide trajectory generation
    head = np.load(config.data.eval_head_path, allow_pickle=True)
    head = torch.from_numpy(head).float()  # [N, 8]

    dataset = TensorDataset(traj, head)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # load statistics
    statistics_path = args.input_dir / args.city / "stats.json"
    _, _, geo_mean, geo_std, len_mean, len_std = load_statistics(statistics_path)

    all_label_traj, all_gen_trajs = [], []
    for batch_traj, batch_head in tqdm(dataloader):
        batch_size = batch_head.size(0)

        lengths = (batch_head[:, 3] * len_std + len_mean).int()
        batch_head = batch_head.cuda()

        # Start with random noise
        x = torch.randn(batch_size, 2, config.data.traj_length).cuda()
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((batch_size,), i, device="cuda", dtype=torch.long)
            next_t = torch.full((batch_size,), j, device="cuda", dtype=torch.long)

            with torch.no_grad():
                pred_noise = unet(x, t, batch_head)
                x = p_xt(x, pred_noise, t, next_t, beta, eta)

        gen_trajs = x.cpu().numpy()

        # resample the trajectory length
        for j in range(batch_size):
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
