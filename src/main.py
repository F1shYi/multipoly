from train.learner import Learner

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    learner = Learner(args.output_dir)
    learner.train()