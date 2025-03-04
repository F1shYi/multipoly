from train.learner import Learner
import argparse
import os
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--config", type=str, required=False, help="Directory to save config files.", default="src/configs/params_v4.yaml")
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    learner = Learner(config)
    learner.train()


    

    # learner = Learner(args.output_dir)
    # learner.train()