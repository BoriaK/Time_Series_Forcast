"""
Start training and verify GPU is active
"""
import subprocess
import sys
import time

print("="*70)
print("STARTING CONVLSTM TRAINING WITH GPU")
print("="*70)

# Verify GPU before starting
import torch
print(f"\nGPU Check:")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Count: {torch.cuda.device_count()}")
else:
    print("  WARNING: No GPU detected!")
    sys.exit(1)

print(f"\nStarting training process...")
print(f"Configuration: d=2 (fixed), 100 epochs, hidden_channels=48")
print("="*70 + "\n")

# Start training
process = subprocess.Popen(
    [sys.executable, "train.py", "--cfg", "configs/cfg_convlstm.yml"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

print(f"Training PID: {process.pid}\n")

# Monitor output
line_count = 0
try:
    while True:
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            time.sleep(0.1)
            continue

        print(line.rstrip())
        line_count += 1

        # Show first 100 lines to confirm it's working
        if line_count >= 100:
            print("\n" + "="*70)
            print("Training is running successfully!")
            print("Switch to simple_monitor.py for progress tracking")
            print("="*70)
            break

except KeyboardInterrupt:
    print("\n\nStopping training...")
    process.terminate()
    process.wait()
