"""
Automated Training Monitor with Parameter Adjustment
Monitors ConvLSTM training and adjusts parameters if overfitting is detected
"""
import time
import yaml
from pathlib import Path
import subprocess
import sys

class TrainingMonitor:
    def __init__(self, config_path="configs/cfg_convlstm.yml", output_dir="outputs/convlstm"):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.args_file = self.output_dir / "args.yml"

        self.best_loss_history = []
        self.train_loss_history = []
        self.overfitting_detected = False

    def load_config(self):
        """Load current configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.load(f, Loader=yaml.Loader)

    def save_config(self, config):
        """Save updated configuration"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

    def check_training_status(self):
        """Check current training status from args.yml"""
        if not self.args_file.exists():
            return None

        with open(self.args_file, 'r') as f:
            args = yaml.load(f, Loader=yaml.Loader)

        return args

    def detect_overfitting(self, train_loss, test_loss):
        """
        Detect overfitting by comparing train and test loss
        Returns True if overfitting is detected
        """
        if train_loss is None or test_loss is None:
            return False

        # Overfitting indicators:
        # 1. Test loss > train loss by significant margin
        gap = test_loss - train_loss
        ratio = test_loss / train_loss if train_loss > 0 else float('inf')

        if gap > 0.1 or ratio > 1.2:  # Test loss is 20% higher than train loss
            return True

        return False

    def detect_no_improvement(self, best_loss_history, window=5):
        """
        Detect if training has stagnated (no improvement in last N epochs)
        """
        if len(best_loss_history) < window:
            return False

        recent = best_loss_history[-window:]
        if all(abs(recent[i] - recent[0]) < 0.001 for i in range(len(recent))):
            return True

        return False

    def adjust_parameters(self, reason="overfitting"):
        """
        Adjust hyperparameters based on detected issue
        """
        config = self.load_config()

        print(f"\n{'='*60}")
        print(f"ADJUSTMENT NEEDED: {reason}")
        print(f"{'='*60}")

        if reason == "overfitting":
            # Increase regularization
            old_wd = config.get('wd', 0.0001)
            new_wd = min(old_wd * 2, 0.001)  # Double weight decay, cap at 0.001
            config['wd'] = new_wd

            # Reduce model capacity
            old_hidden = config.get('hidden_channels', 64)
            new_hidden = max(int(old_hidden * 0.75), 32)  # Reduce by 25%, min 32
            config['hidden_channels'] = new_hidden

            # Reduce learning rate
            old_lr = config.get('max_lr', 0.001)
            new_lr = old_lr * 0.8
            config['max_lr'] = new_lr

            print(f"Adjustments made:")
            print(f"  - Weight decay: {old_wd} → {new_wd}")
            print(f"  - Hidden channels: {old_hidden} → {new_hidden}")
            print(f"  - Learning rate: {old_lr} → {new_lr}")

        elif reason == "no_improvement":
            # Try increasing learning rate slightly
            old_lr = config.get('max_lr', 0.001)
            new_lr = old_lr * 1.2
            config['max_lr'] = min(new_lr, 0.002)  # Cap at 0.002

            print(f"Adjustments made:")
            print(f"  - Learning rate: {old_lr} → {config['max_lr']}")

        self.save_config(config)
        print(f"Configuration saved to {self.config_path}")
        print(f"{'='*60}\n")

        return config

    def print_status(self, args, check_num):
        """Print current training status"""
        print(f"\n{'='*60}")
        print(f"Training Status Check #{check_num}")
        print(f"{'='*60}")

        if args is None:
            print("Training not started yet or args.yml not found")
            return

        best_epoch = args.get('best_epoch', 'N/A')
        best_loss = args.get('best_loss', None)
        last_epoch = args.get('last_epoch', 'N/A')

        print(f"Best Epoch: {best_epoch}")
        if best_loss is not None:
            print(f"Best Loss: {best_loss:.6f}")
        print(f"Last Epoch: {last_epoch}")

        # Model parameters
        print(f"\nModel Configuration:")
        print(f"  Hidden Channels: {args.get('hidden_channels', 'N/A')}")
        print(f"  Num Layers: {args.get('num_layers', 'N/A')}")
        print(f"  Kernel Size: {args.get('kernel_size', 'N/A')}")
        print(f"  Learning Rate: {args.get('max_lr', 'N/A')}")
        print(f"  Weight Decay: {args.get('wd', 'N/A')}")

        print(f"{'='*60}\n")

    def monitor(self, check_interval=30, max_checks=40):
        """
        Main monitoring loop
        """
        print("\n" + "="*60)
        print("ConvLSTM Training Monitor Started")
        print(f"Check interval: {check_interval}s")
        print(f"Maximum checks: {max_checks}")
        print("="*60)

        for check in range(1, max_checks + 1):
            time.sleep(check_interval)

            args = self.check_training_status()
            self.print_status(args, check)

            if args and 'best_loss' in args:
                best_loss = args['best_loss']
                self.best_loss_history.append(best_loss)

                # Check for stagnation
                if self.detect_no_improvement(self.best_loss_history):
                    print("⚠ WARNING: No improvement detected in last 5 checks")
                    # Optionally adjust parameters
                    # self.adjust_parameters(reason="no_improvement")

                # Check if we have enough history to evaluate
                if len(self.best_loss_history) >= 3:
                    recent_best = self.best_loss_history[-1]
                    improvement = self.best_loss_history[0] - recent_best
                    improvement_pct = (improvement / self.best_loss_history[0]) * 100 if self.best_loss_history[0] > 0 else 0

                    print(f"Improvement so far: {improvement:.6f} ({improvement_pct:.2f}%)")

                    if improvement < 0:
                        print("⚠ WARNING: Test loss is increasing!")

        print("\n" + "="*60)
        print("Monitoring Complete")
        print("="*60)

        # Final summary
        if len(self.best_loss_history) > 0:
            print(f"\nFinal Summary:")
            print(f"  Initial best loss: {self.best_loss_history[0]:.6f}")
            print(f"  Final best loss: {self.best_loss_history[-1]:.6f}")
            print(f"  Total improvement: {self.best_loss_history[0] - self.best_loss_history[-1]:.6f}")


if __name__ == "__main__":
    monitor = TrainingMonitor()

    # Check every 30 seconds, up to 40 times (20 minutes total)
    monitor.monitor(check_interval=30, max_checks=40)
