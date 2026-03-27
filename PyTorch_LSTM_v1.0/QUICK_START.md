# ConvLSTM Training - Quick Start Guide

## How to Run Training

### Option 1: Training WITH Parameter Optimization (Recommended)
```bash
cd C:\Users\Administrator\Python_Projects\Time_Series_Forcast\PyTorch_LSTM_v1.0
python train.py --cfg configs/cfg_convlstm.yml --optimize
```

**Features enabled:**
- Auto-adjusts parameters on early stopping
- Updates original config on successful completion
- Self-optimizing across training runs

### Option 2: Training WITHOUT Parameter Optimization
```bash
cd C:\Users\Administrator\Python_Projects\Time_Series_Forcast\PyTorch_LSTM_v1.0
python train.py --cfg configs/cfg_convlstm.yml
```

**Features enabled:**
- GPU acceleration
- Early stopping (10 epochs patience)
- Best model saving
- Moving average tracking

**Features disabled:**
- No adjusted config creation on failure
- No automatic config updates on success
- Manual parameter tuning required

---

## What Happens With --optimize Flag

### ✅ Always Active (with or without --optimize):
1. **GPU acceleration** - Automatically uses NVIDIA RTX 4060
2. **Test loss tracking** - Monitors with 5-epoch moving average
3. **Early stopping** - Stops if no improvement for 10 epochs
4. **Best model saving** - Saves checkpoint when loss improves
5. **Spike detection** - Ignores single-iteration noise

### ✅ Only Active WITH --optimize Flag:

#### If Training is Unsuccessful (Early Stopping Triggered):
- **Adjusted config created** - `outputs/convlstm/cfg_convlstm_adjusted.yml` saved with improved parameters
- **Next run** - Use: `python train.py --cfg outputs/convlstm/cfg_convlstm_adjusted.yml --optimize`

#### If Training Completes Successfully (All 50 Epochs):
- **Original config updated** - `configs/cfg_convlstm.yml` updated with parameters from adjusted config (if exists)
- **Adjusted config removed** - No longer needed since parameters are now in original
- **Next run automatic** - Run `python train.py --cfg configs/cfg_convlstm.yml --optimize` with optimized settings

### ❌ WITHOUT --optimize Flag:
- **No parameter adjustment** - Original config never modified
- **No adjusted config created** - Manual parameter tuning required
- **Simple training only** - Train, evaluate, save checkpoints

---

## Monitoring Progress

### Option 1: Check Status Anytime
```bash
python quick_status.py
```

### Option 2: View Files Directly
```bash
# Current status
Get-Content outputs/convlstm/args.yml

# List checkpoints
Get-ChildItem outputs/convlstm/*.pt
```

---

## Output Files

After training completes:
```
configs/
└── cfg_convlstm.yml          # Auto-updated if training succeeded

outputs/convlstm/
├── chkpnt_convlstm_Best_epoch_X.pt      # Best model
├── chkpnt_convlstm_Last_epoch_50.pt     # Final model
├── args.yml                              # Training results
└── cfg_convlstm_adjusted.yml            # Only if early stopping triggered
```

---

## That's All!

Just run:
```bash
python train.py --cfg configs/cfg_convlstm.yml
```

The system is fully autonomous and self-optimizing.

---

## Recent Training Results

Your last training:
- **Best Loss:** 0.003918 (99.15% improvement from start)
- **Best Epoch:** 43
- **Final Epoch:** 50 (completed successfully)
- **Config auto-updated** with evolved parameters (hidden: 75, kernel: 7)

Next training will automatically use these optimized settings!
