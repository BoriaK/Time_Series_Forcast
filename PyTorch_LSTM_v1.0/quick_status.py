"""
Simple status check for ConvLSTM training
"""
import yaml
from pathlib import Path
import torch

checkpoint_dir = Path("outputs/convlstm")
args_file = checkpoint_dir / "args.yml"

print("="*70)
print("CONVLSTM TRAINING STATUS")
print("="*70)

if args_file.exists():
    with open(args_file, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)

    print(f"\n✓ Training is RUNNING")
    print(f"\nCurrent Status:")
    print(f"  Best Epoch: {args.get('best_epoch', 'N/A')}")
    print(f"  Configuration:")
    print(f"    - Hidden channels: {args.get('hidden_channels', 'N/A')}")
    print(f"    - Num layers: {args.get('num_layers', 'N/A')}")
    print(f"    - Kernel size: {args.get('kernel_size', 'N/A')}")
    print(f"    - Learning rate: {args.get('max_lr', 'N/A')}")
    print(f"    - Weight decay: {args.get('wd', 'N/A')}")
    print(f"    - d value: {args.get('d', 'N/A')}")

    # Check for checkpoints
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    print(f"\n  Checkpoints saved: {len(checkpoints)}")

    if checkpoints:
        print(f"\n  Checkpoint Details:")
        for ckpt_file in sorted(checkpoints):
            try:
                ckpt = torch.load(ckpt_file, map_location='cpu')
                epoch_num = ckpt_file.stem.split('_')[-1]
                best_loss = ckpt.get('best_loss', 'N/A')
                print(f"    - Epoch {epoch_num}: Loss = {best_loss}")
            except:
                print(f"    - {ckpt_file.name}: Could not load")
else:
    print("\n⚠️  Training has not started or args.yml not created yet")

print("="*70)
