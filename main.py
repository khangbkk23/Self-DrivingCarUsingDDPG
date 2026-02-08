import yaml
import os
from ddpg.training.run_train import run_train

def main():
    config_path = './config/env.yaml'
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"Starting Training: {cfg['env']}")
    run_train(cfg)

if __name__ == "__main__":
    main()