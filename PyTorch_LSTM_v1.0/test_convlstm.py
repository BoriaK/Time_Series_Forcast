"""
Test script for ConvLSTM model
"""
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from modules import ConvLSTM, LSTMO, CNNLSTM

def test_convlstm():
    print("=" * 60)
    print("Testing ConvLSTM Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model parameters
    batch_size = 4
    input_channels = 1
    seq_len = 256
    hidden_channels = 64
    num_layers = 2
    kernel_size = 3

    print(f"\nModel Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Input channels: {input_channels}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Kernel size: {kernel_size}")

    # Create model
    model = ConvLSTM(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        device=device
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create dummy input
    x = torch.randn(batch_size, input_channels, seq_len).to(device)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 1)")

    # Check output shape
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    print("\n✓ ConvLSTM test passed!")

    # Test with different batch sizes
    print("\nTesting with different batch sizes...")
    for bs in [1, 8, 16]:
        x_test = torch.randn(bs, input_channels, seq_len).to(device)
        with torch.no_grad():
            out_test = model(x_test)
        assert out_test.shape == (bs, 1)
        print(f"  Batch size {bs}: ✓")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def compare_models():
    """Compare ConvLSTM with LSTMO and CNNLSTM"""
    print("\n" + "=" * 60)
    print("Comparing Models")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    seq_len = 256

    models = {
        'LSTMO': LSTMO(device=device),
        'CNNLSTM': CNNLSTM(device=device),
        'ConvLSTM': ConvLSTM(input_channels=1, hidden_channels=64, num_layers=2, kernel_size=3, device=device)
    }

    x = torch.randn(batch_size, 1, seq_len).to(device)

    print(f"\nModel Comparison (input: {x.shape}):\n")
    print(f"{'Model':<15} {'Parameters':<15} {'Output Shape':<20}")
    print("-" * 50)

    for name, model in models.items():
        model = model.to(device).eval()
        params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            output = model(x)

        print(f"{name:<15} {params:<15,} {str(output.shape):<20}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_convlstm()
    compare_models()
