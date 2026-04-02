import json
from argparse import ArgumentParser
from ast import literal_eval
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector
from tqdm.contrib.concurrent import process_map


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=Path("data/"))
    parser.add_argument("--city", type=str, choices=["Beijing", "Porto", "San_Francisco"])
    parser.add_argument("--trajectory_length", type=int, default=200)
    return parser.parse_args()


def resample_trajectory(x, length):
    """
    Resample a trajectory to a fixed length using linear interpolation
    :param x: trajectory to resample
    :param length: length of the resampled trajectory
    :return: resampled trajectory
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T


def calculate_grid_id(
    lat: float, lon: float, min_lat: float, max_lat: float, min_lon: float, max_lon: float, grid_divisions: int = 16
):
    lat_idx = int((lat - min_lat) / (max_lat - min_lat) * (grid_divisions - 1))
    lon_idx = int((lon - min_lon) / (max_lon - min_lon) * (grid_divisions - 1))

    grid_id = lat_idx * grid_divisions + lon_idx
    return grid_id


def calculate_trajectory_attributes(
    trajectory: np.ndarray, time_list: list, min_lat: float, max_lat: float, min_lon: float, max_lon: float
):
    trip_length = trajectory.shape[0]
    points_a = np.flip(trajectory[:-1], axis=1)  # flip to (lat, lon)
    points_b = np.flip(trajectory[1:], axis=1)
    trip_distance_m = haversine_vector(points_a, points_b, unit=Unit.METERS).sum()

    avg_dis_m = trip_distance_m / (trip_length - 1) if trip_length > 1 else 0.0
    trip_time_min = (time_list[-1] - time_list[0]).total_seconds() / 60.0 if len(time_list) > 1 else 0.0
    avg_speed_m_s = trip_distance_m / (trip_time_min * 60) if trip_time_min > 0 else 0.0

    departure_time = (time_list[0].hour * 60 + time_list[0].minute) // 5

    grid_fn = lambda lat, lon: calculate_grid_id(lat, lon, min_lat, max_lat, min_lon, max_lon)
    start_id = grid_fn(trajectory[0][0], trajectory[0][1])
    end_id = grid_fn(trajectory[-1][0], trajectory[-1][1])

    return {
        "departure_time_h": departure_time,
        "trip_distance_m": trip_distance_m,
        "trip_time_min": trip_time_min,
        "trip_length": trip_length,
        "avg_dis_m": avg_dis_m,
        "avg_speed_m_s": avg_speed_m_s,
        "start_id": start_id,
        "end_id": end_id,
    }


def process_row(
    args,
    geo2coords: dict[int, list],
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    trajectory_length: int,
) -> tuple[np.ndarray, dict, int]:
    _, row = args
    rid_list = row["rid_list"]
    time_list = row["time_list"]
    rid_list = literal_eval(row["rid_list"])
    time_list_str = row["time_list"].split(",")
    time_list = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in time_list_str]

    geometry_list = [geo2coords[rid] for rid in rid_list]
    geometry_list = [g for geo in geometry_list for g in geo]
    trajectory = np.array(geometry_list)  # [L, 2] (lon, lat)
    resampled_trajectory = resample_trajectory(trajectory, length=trajectory_length)
    attrs = calculate_trajectory_attributes(trajectory, time_list, min_lat, max_lat, min_lon, max_lon)
    trajectory_lengh = len(trajectory)
    return resampled_trajectory, attrs, trajectory_lengh


def main(args):
    train_file = pd.read_csv(args.input_dir / args.city / "train.csv")
    test_file = pd.read_csv(args.input_dir / args.city / "test.csv")
    val_file = pd.read_csv(args.input_dir / args.city / "val.csv")
    geo_file = args.input_dir / args.city / "roadmap.geo"

    geo = pd.read_csv(geo_file)
    geo2coords = {rid: eval(coords) for rid, coords in zip(geo["geo_id"], geo["coordinates"])}

    min_lat = min([c[0] for coord in geo2coords.values() for c in coord])
    max_lat = max([c[0] for coord in geo2coords.values() for c in coord])
    min_lon = min([c[1] for coord in geo2coords.values() for c in coord])
    max_lon = max([c[1] for coord in geo2coords.values() for c in coord])

    fn = partial(
        process_row,
        geo2coords=geo2coords,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        trajectory_length=args.trajectory_length,
    )
    train_results = process_map(fn, train_file.iterrows(), max_workers=32, total=len(train_file), chunksize=1000)
    test_results = process_map(fn, test_file.iterrows(), max_workers=32, total=len(test_file), chunksize=1000)
    val_results = process_map(fn, val_file.iterrows(), max_workers=32, total=len(val_file), chunksize=1000)

    train_trajectories, train_attrs, train_trajectory_lengths = zip(*train_results)
    test_trajectories, test_attrs, _ = zip(*test_results)
    val_trajectories, val_attrs, _ = zip(*val_results)

    # calculate mean and std of continuous attributes
    attr_df = pd.DataFrame(train_attrs)
    cont_cols = [c for c in attr_df.columns if c not in ["start_id", "end_id", "departure_time_h"]]
    train_stats = attr_df[cont_cols].agg(["mean", "std"]).to_dict()

    # calculate mean and std of trajectory coordinates and trajectory lengths
    train_trajectories = np.array(train_trajectories)  # [N, L, 2]
    test_trajectories = np.array(test_trajectories)
    val_trajectories = np.array(val_trajectories)

    trajectory_mean = train_trajectories.mean(axis=(0, 1))
    trajectory_std = train_trajectories.std(axis=(0, 1))
    train_stats["lon"] = {"mean": trajectory_mean[0], "std": trajectory_std[0]}
    train_stats["lat"] = {"mean": trajectory_mean[1], "std": trajectory_std[1]}
    train_stats["traj_len"] = {"mean": np.mean(train_trajectory_lengths), "std": np.std(train_trajectory_lengths)}

    # normalize continuous attributes using train stats, and convert to numpy arrays
    def normalize_attrs(attrs: list[dict]) -> np.ndarray:
        attrs_df = pd.DataFrame(attrs)
        attrs_df[cont_cols] = (attrs_df[cont_cols] - attrs_df[cont_cols].mean()) / attrs_df[cont_cols].std()
        return attrs_df.to_numpy()

    train_attrs_array = normalize_attrs(train_attrs)
    test_attrs_array = normalize_attrs(test_attrs)
    val_attrs_array = normalize_attrs(val_attrs)

    # normalize trajectory coordinates using train stats
    traj_means = np.array([train_stats["lon"]["mean"], train_stats["lat"]["mean"]])
    traj_stds = np.array([train_stats["lon"]["std"], train_stats["lat"]["std"]])

    def normalize_trajectories(trajectories: np.ndarray) -> np.ndarray:
        return (trajectories - traj_means) / traj_stds

    train_trajectories = normalize_trajectories(train_trajectories)
    test_trajectories = normalize_trajectories(test_trajectories)
    val_trajectories = normalize_trajectories(val_trajectories)

    assert train_trajectories.shape == (len(train_file), args.trajectory_length, 2)
    assert test_trajectories.shape == (len(test_file), args.trajectory_length, 2)
    assert val_trajectories.shape == (len(val_file), args.trajectory_length, 2)
    assert train_attrs_array.shape == (len(train_file), len(train_attrs[0]))
    assert test_attrs_array.shape == (len(test_file), len(test_attrs[0]))
    assert val_attrs_array.shape == (len(val_file), len(val_attrs[0]))

    # export train stats, trajectories and attributes to files
    with open(args.input_dir / args.city / "stats.json", "w") as f:
        json.dump(train_stats, f)

    np.save(args.input_dir / args.city / "train_trajectories.npy", train_trajectories)
    np.save(args.input_dir / args.city / "test_trajectories.npy", test_trajectories)
    np.save(args.input_dir / args.city / "val_trajectories.npy", val_trajectories)

    np.save(args.input_dir / args.city / "train_attrs.npy", train_attrs_array)
    np.save(args.input_dir / args.city / "test_attrs.npy", test_attrs_array)
    np.save(args.input_dir / args.city / "val_attrs.npy", val_attrs_array)


if __name__ == "__main__":
    args = parse_args()
    main(args)
