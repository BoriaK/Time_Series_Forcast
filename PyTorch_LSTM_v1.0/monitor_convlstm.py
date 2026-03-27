"""
Real-time Training Monitor for ConvLSTM
Monitors test loss progression and detects if network is learning
Tracks overall trend, not just best loss
"""
import time
import yaml
from pathlib import Path
import sys
import torch

def moving_average(data, window=5):
    """Calculate moving average"""
    if len(data) < window:
        return sum(data) / len(data) if data else 0
    return sum(data[-window:]) / window

def monitor_training():
    checkpoint_dir = Path("outputs/convlstm")
    args_file = checkpoint_dir / "args.yml"

    print("="*70)
    print("CONVLSTM TRAINING MONITOR - TRACKING TEST LOSS PROGRESSION")
    print("="*70)
    print("Configuration: d=[0.2], 50 epochs max")
    print("Early stopping: 10 evaluations without real improvement")
    print("Monitors: Overall trend (not just single best)")
    print("="*70)
    print("\nWaiting for training to start...\n")

    test_loss_history = []
    checkpoint_losses = {}  # epoch -> loss
    check_count = 0
    last_checkpoint_count = 0

    while check_count < 60:  # Monitor for up to 20 minutes
        time.sleep(20)
        check_count += 1

        if not args_file.exists():
            print(f"[Check {check_count:2d}] Waiting for training initialization...")
            continue

        try:
            # Load saved checkpoints and extract losses
            checkpoint_files = list(checkpoint_dir.glob("chkpnt_convlstm_Best_epoch_*.pt"))

            if len(checkpoint_files) > last_checkpoint_count:
                # New checkpoint saved - load and extract loss
                for ckpt_file in checkpoint_files:
                    epoch_num = int(ckpt_file.stem.split('_')[-1])
                    if epoch_num not in checkpoint_losses:
                        try:
                            ckpt = torch.load(ckpt_file, map_location='cpu')
                            if 'best_loss' in ckpt:
                                checkpoint_losses[epoch_num] = ckpt['best_loss']
                                test_loss_history.append(ckpt['best_loss'])
                        except:
                            pass

                last_checkpoint_count = len(checkpoint_files)

                if test_loss_history:
                    current_loss = test_loss_history[-1]

                    print(f"\n{'='*70}")
                    print(f"📊 NEW EVALUATION - Check #{check_count}")
                    print(f"{'='*70}")
                    print(f"  Current test loss: {current_loss:.8f}")

                    if len(test_loss_history) >= 2:
                        prev_loss = test_loss_history[-2]
                        change = prev_loss - current_loss
                        change_pct = (change / prev_loss) * 100

                        if change > 0:
                            print(f"  Change from prev: ↓ {change:.8f} ({change_pct:.2f}%) ✓")
                        else:
                            print(f"  Change from prev: ↑ {abs(change):.8f} ({abs(change_pct):.2f}%) ⚠️")

                    # Show moving average
                    if len(test_loss_history) >= 3:
                        ma3 = moving_average(test_loss_history, 3)
                        ma5 = moving_average(test_loss_history, 5)
                        print(f"  Moving Avg (3): {ma3:.8f}")
                        print(f"  Moving Avg (5): {ma5:.8f}")

                        # Trend analysis
                        if len(test_loss_history) >= 5:
                            recent_avg = moving_average(test_loss_history[-3:], 3)
                            earlier_avg = moving_average(test_loss_history[-6:-3], 3) if len(test_loss_history) >= 6 else test_loss_history[0]
                            trend = earlier_avg - recent_avg

                            if trend > 0.001:
                                print(f"  Trend: ✓ IMPROVING (Δ={trend:.8f})")
                            elif trend < -0.001:
                                print(f"  Trend: ⚠️  DEGRADING (Δ={trend:.8f})")
                            else:
                                print(f"  Trend: ⚠️  STAGNANT (Δ={trend:.8f})")

                    # Show history
                    print(f"\n  Loss history ({len(test_loss_history)} evals):")
                    print(f"    {[f'{l:.6f}' for l in test_loss_history[-10:]]}")
                    print(f"{'='*70}\n")
            else:
                print(f"[Check {check_count:2d}] Training in progress... ({len(test_loss_history)} evaluations recorded)")

        except Exception as e:
            print(f"[Check {check_count:2d}] Error: {e}")

    print("\n" + "="*70)
    print("MONITORING SESSION COMPLETE")
    print("="*70)

    if len(test_loss_history) > 0:
        print(f"\n📊 FINAL ANALYSIS:")
        print(f"  Total evaluations: {len(test_loss_history)}")
        print(f"  First test loss: {test_loss_history[0]:.8f}")
        print(f"  Final test loss: {test_loss_history[-1]:.8f}")
        print(f"  Best test loss: {min(test_loss_history):.8f}")

        total_improvement = test_loss_history[0] - test_loss_history[-1]
        total_improvement_pct = (total_improvement / test_loss_history[0]) * 100

        print(f"\n  Overall change: {total_improvement:.8f} ({total_improvement_pct:.2f}%)")

        # Final assessment
        if len(test_loss_history) >= 5:
            ma_recent = moving_average(test_loss_history[-5:], 5)
            ma_first = moving_average(test_loss_history[:5], 5) if len(test_loss_history) >= 5 else test_loss_history[0]
            ma_improvement = ma_first - ma_recent

            print(f"  MA improvement: {ma_improvement:.8f}")

            if ma_improvement > 0.005:
                print(f"\n  ✓✓ VERDICT: Network IS LEARNING")
            elif ma_improvement > 0.001:
                print(f"\n  ✓ VERDICT: Network is learning (slowly)")
            elif ma_improvement > 0:
                print(f"\n  ⚠️  VERDICT: Marginal learning - consider parameter adjustment")
            else:
                print(f"\n  ❌ VERDICT: Network is NOT learning - MUST adjust parameters")

        print(f"\n  Full loss history:")
        print(f"    {[f'{l:.6f}' for l in test_loss_history]}")

    print("="*70)

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")
        sys.exit(0)
