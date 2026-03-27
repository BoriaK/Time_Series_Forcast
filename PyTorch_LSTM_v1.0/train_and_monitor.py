"""
Simple training launcher and monitor
"""
import subprocess
import time
import yaml
from pathlib import Path

def monitor_training(max_wait=120):
    """Monitor training by checking args.yml"""
    output_file = Path("outputs/convlstm/args.yml")

    print("="*60)
    print("Starting ConvLSTM Training...")
    print("="*60)

    # Start training process
    process = subprocess.Popen(
        ["python", "train.py", "--cfg", "configs/cfg_convlstm.yml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    print("\nTraining process started (PID: {})".format(process.pid))
    print("Monitoring progress...\n")

    start_time = time.time()
    last_best_loss = None
    last_best_epoch = None
    checks = 0

    try:
        while True:
            # Read output line by line
            line = process.stdout.readline()
            if line:
                print(line.strip())

                # Check for loss values
                if "test: loss" in line:
                    print("\n" + "="*60)
                    print("CHECKPOINT REACHED")
                    print(line.strip())
                    print("="*60 + "\n")

            # Check if process has finished
            if process.poll() is not None:
                print("\nTraining process completed")
                break

            # Periodic status check
            if int(time.time() - start_time) % 30 == 0:
                checks += 1
                if output_file.exists():
                    with open(output_file, 'r') as f:
                        args = yaml.load(f, Loader=yaml.Loader)
                        best_epoch = args.get('best_epoch', 'N/A')
                        best_loss = args.get('best_loss', None)

                        if best_loss != last_best_loss:
                            print(f"\n[Check #{checks}] Best epoch: {best_epoch}, Best loss: {best_loss:.6f}")
                            last_best_loss = best_loss
                            last_best_epoch = best_epoch

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        process.terminate()
        process.wait()

    return last_best_loss, last_best_epoch

if __name__ == "__main__":
    best_loss, best_epoch = monitor_training()
    print(f"\n\nFinal Results:")
    print(f"Best Loss: {best_loss}")
    print(f"Best Epoch: {best_epoch}")
