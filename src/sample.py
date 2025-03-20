from sampler.eight_bar import EightBarSampler
from sampler.whole_song import WholeSongSampler
import argparse
import os
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Directory to save config files.",
        default="src/configs/whole_song_sample.yaml",
    )
    parser.add_argument("--mode", type=str, required=False, default="8bar")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "8bar":
        sampler = EightBarSampler(config)
        sampler.run()
    else:
        sampler = WholeSongSampler(config)
        sampler.run()
