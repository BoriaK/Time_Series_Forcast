"""
Simple Training Monitor - Checks progress every 20 seconds
"""
import time
import yaml
from pathlib import Path

def monitor():
    args_file = Path("outputs/convlstm/args.yml")

    print("="*70)
    print("CONVLSTM TRAINING MONITOR - Configuration: d=2 (fixed)")
    print("="*70)
    print("\nWaiting for training to start...\n")

    last_best_loss = None
    last_best_epoch = None
    check_count = 0
    loss_history = []

    while check_count < 30:  # Monitor for up to 10 minutes (30 * 20 seconds)
        time.sleep(20)
        check_count += 1

        if not args_file.exists():
            print(f"[Check {check_count}] Waiting for training to initialize...")
            continue

        try:
            with open(args_file, 'r') as f:
                args = yaml.load(f, Loader=yaml.Loader)

            best_epoch = args.get('best_epoch', None)
            best_loss = args.get('best_loss', None)

            if best_loss is not None:
                loss_history.append(best_loss)

                if best_loss != last_best_loss:
                    print(f"\n{'='*70}")
                    print(f"[Check {check_count}] PROGRESS UPDATE")
                    print(f"{'='*70}")
                    print(f"  Best Epoch: {best_epoch}")
                    print(f"  Best Loss:  {best_loss:.8f}")

                    if last_best_loss is not None:
                        improvement = last_best_loss - best_loss
                        improvement_pct = (improvement / last_best_loss) * 100
                        print(f"  Improvement: {improvement:.8f} ({improvement_pct:.2f}%)")

                        if improvement < 0:
                            print(f"  ⚠️  WARNING: Loss increased!")
                        elif improvement < 0.0001:
                            print(f"  ⚠️  WARNING: Very small improvement - may be stuck")

                    # Model info
                    print(f"\n  Model Config:")
                    print(f"    - Hidden channels: {args.get('hidden_channels', 'N/A')}")
                    print(f"    - Num layers: {args.get('num_layers', 'N/A')}")
                    print(f"    - Kernel size: {args.get('kernel_size', 'N/A')}")
                    print(f"    - Learning rate: {args.get('max_lr', 'N/A')}")
                    print(f"    - Weight decay: {args.get('wd', 'N/A')}")
                    print(f"    - d value: {args.get('d', 'N/A')}")
                    print(f"{'='*70}\n")

                    last_best_loss = best_loss
                    last_best_epoch = best_epoch
                else:
                    print(f"[Check {check_count}] No improvement yet... (Best: {best_loss:.8f} at epoch {best_epoch})")

                # Check for no improvement
                if len(loss_history) >= 5:
                    recent_5 = loss_history[-5:]
                    if all(abs(recent_5[i] - recent_5[0]) < 0.00001 for i in range(len(recent_5))):
                        print(f"\n⚠️  WARNING: No improvement in last 5 checks - model may not be learning!")
                        print(f"   Consider stopping and adjusting hyperparameters\n")

        except Exception as e:
            print(f"[Check {check_count}] Error reading args: {e}")

    print("\n" + "="*70)
    print("MONITORING COMPLETE")
    print("="*70)

    if last_best_loss:
        print(f"\nFinal Best Loss: {last_best_loss:.8f} at epoch {last_best_epoch}")
        if len(loss_history) >= 2:
            total_improvement = loss_history[0] - loss_history[-1]
            print(f"Total Improvement: {total_improvement:.8f}")

if __name__ == "__main__":
    monitor()
