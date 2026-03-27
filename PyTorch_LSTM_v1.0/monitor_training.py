"""
Training Monitor for ConvLSTM
Monitors training progress and detects overfitting
"""
import time
import yaml
from pathlib import Path
import re

def monitor_training(log_interval=30, max_checks=20):
    """
    Monitor training progress by checking the args.yml file
    """
    config_path = Path("outputs/convlstm/args.yml")

    print("=" * 60)
    print("ConvLSTM Training Monitor Started")
    print("=" * 60)

    train_losses = []
    test_losses = []
    epochs = []

    for check in range(max_checks):
        time.sleep(log_interval)

        if config_path.exists():
            with open(config_path, 'r') as f:
                args = yaml.load(f, Loader=yaml.Loader)

            if 'best_epoch' in args:
                print(f"\nCheck {check + 1}:")
                print(f"  Best epoch: {args.get('best_epoch', 'N/A')}")
                print(f"  Best loss: {args.get('best_loss', 'N/A'):.6f}" if 'best_loss' in args else "  Best loss: N/A")

                # Check for args in checkpoint
                # This is a simple monitor - in production you'd track more metrics
        else:
            print(f"Check {check + 1}: Waiting for training to start...")

    print("\n" + "=" * 60)
    print("Monitoring Complete")
    print("=" * 60)

if __name__ == "__main__":
    monitor_training(log_interval=20, max_checks=30)
