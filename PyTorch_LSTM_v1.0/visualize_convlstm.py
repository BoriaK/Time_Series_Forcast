"""
Visualization script for ConvLSTM architecture
Shows the difference between Standard LSTM and ConvLSTM
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def visualize_convlstm_cell():
    """Visualize the ConvLSTM cell architecture"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Standard LSTM Cell
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Standard LSTM Cell', fontsize=16, fontweight='bold')

    # ConvLSTM Cell
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('ConvLSTM Cell (Ours)', fontsize=16, fontweight='bold')

    # Add gates for Standard LSTM
    gate_y = 5
    gate_x = [2, 4, 6, 8]
    gate_names = ['Input\nGate', 'Forget\nGate', 'Cell\nGate', 'Output\nGate']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    for i, (x, name, color) in enumerate(zip(gate_x, gate_names, colors)):
        # Standard LSTM - FC operation
        rect = FancyBboxPatch((x-0.4, gate_y-0.4), 0.8, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, gate_y, 'FC', ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(x, gate_y-1.2, name, ha='center', va='center', fontsize=9)

        # ConvLSTM - Conv operation
        rect = FancyBboxPatch((x-0.4, gate_y-0.4), 0.8, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x, gate_y, 'Conv', ha='center', va='center', fontsize=9, fontweight='bold')
        ax2.text(x, gate_y-1.2, name, ha='center', va='center', fontsize=9)

    # Input and hidden state
    for ax, title in [(ax1, 'Standard'), (ax2, 'ConvLSTM')]:
        # Input x_t
        ax.add_patch(FancyBboxPatch((0.5, 8), 1.5, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor='lightblue', edgecolor='black', linewidth=2))
        ax.text(1.25, 8.3, 'x_t', ha='center', va='center', fontsize=11, fontweight='bold')

        # Hidden state h_{t-1}
        ax.add_patch(FancyBboxPatch((0.5, 7), 1.5, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor='lightgreen', edgecolor='black', linewidth=2))
        ax.text(1.25, 7.3, 'h_{t-1}', ha='center', va='center', fontsize=11, fontweight='bold')

        # Cell state c_{t-1}
        ax.add_patch(FancyBboxPatch((0.5, 1.5), 1.5, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor='lightyellow', edgecolor='black', linewidth=2))
        ax.text(1.25, 1.8, 'c_{t-1}', ha='center', va='center', fontsize=11, fontweight='bold')

        # Output c_t
        ax.add_patch(FancyBboxPatch((8, 1.5), 1.5, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor='lightyellow', edgecolor='black', linewidth=2))
        ax.text(8.75, 1.8, 'c_t', ha='center', va='center', fontsize=11, fontweight='bold')

        # Output h_t
        ax.add_patch(FancyBboxPatch((8, 7), 1.5, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor='lightgreen', edgecolor='black', linewidth=2))
        ax.text(8.75, 7.3, 'h_t', ha='center', va='center', fontsize=11, fontweight='bold')

    # Add operation descriptions
    ax1.text(5, 0.5, 'Operations: Fully Connected (Matrix Multiplication)',
             ha='center', fontsize=10, style='italic')
    ax1.text(5, 0.1, 'Loses spatial/temporal structure',
             ha='center', fontsize=9, style='italic', color='red')

    ax2.text(5, 0.5, 'Operations: Convolution (Preserves Structure)',
             ha='center', fontsize=10, style='italic')
    ax2.text(5, 0.1, 'Maintains spatial/temporal relationships',
             ha='center', fontsize=9, style='italic', color='green')

    plt.tight_layout()
    plt.savefig('convlstm_architecture.png', dpi=300, bbox_inches='tight')
    print("Saved: convlstm_architecture.png")
    plt.close()


def visualize_model_comparison():
    """Visualize the full model architectures"""
    fig = plt.figure(figsize=(18, 6))

    models = ['LSTMO', 'CNNLSTM', 'ConvLSTM']
    params = [199299, 210273, 148803]

    for idx, (model_name, param_count) in enumerate(zip(models, params)):
        ax = plt.subplot(1, 3, idx+1)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        ax.set_title(f'{model_name}\n({param_count:,} parameters)',
                     fontsize=14, fontweight='bold')

        y_pos = 11

        # Input
        ax.add_patch(FancyBboxPatch((3, y_pos-0.5), 4, 0.8,
                                    facecolor='lightblue', edgecolor='black', linewidth=2))
        ax.text(5, y_pos-0.1, 'Input (batch, 1, seq_len)', ha='center', fontsize=10)
        y_pos -= 1.5

        # BatchNorm
        ax.add_patch(FancyBboxPatch((3, y_pos-0.5), 4, 0.8,
                                    facecolor='lightgray', edgecolor='black', linewidth=2))
        ax.text(5, y_pos-0.1, 'BatchNorm1d', ha='center', fontsize=10)
        y_pos -= 1.5

        if model_name == 'LSTMO':
            # LSTM layers
            ax.add_patch(FancyBboxPatch((3, y_pos-0.5), 4, 1.5,
                                        facecolor='#FFE5B4', edgecolor='black', linewidth=2))
            ax.text(5, y_pos, 'LSTM (2 layers)', ha='center', fontsize=10)
            ax.text(5, y_pos-0.4, 'Fully Connected Gates', ha='center', fontsize=8, style='italic')
            y_pos -= 2.5

        elif model_name == 'CNNLSTM':
            # CNN layers
            ax.add_patch(FancyBboxPatch((3, y_pos-0.5), 4, 1.8,
                                        facecolor='#B4E5FF', edgecolor='black', linewidth=2))
            ax.text(5, y_pos+0.2, 'Conv Stack', ha='center', fontsize=10)
            ax.text(5, y_pos-0.3, '+ ResBlocks (dilation 1,3,9)', ha='center', fontsize=8)
            y_pos -= 2.5

            # LSTM layers
            ax.add_patch(FancyBboxPatch((3, y_pos-0.5), 4, 1.2,
                                        facecolor='#FFE5B4', edgecolor='black', linewidth=2))
            ax.text(5, y_pos-0.1, 'LSTM (2 layers)', ha='center', fontsize=10)
            y_pos -= 2

        else:  # ConvLSTM
            # ConvLSTM layers
            ax.add_patch(FancyBboxPatch((3, y_pos-0.5), 4, 1.8,
                                        facecolor='#B4FFB4', edgecolor='black', linewidth=2))
            ax.text(5, y_pos+0.2, 'ConvLSTM (2 layers)', ha='center', fontsize=10)
            ax.text(5, y_pos-0.3, 'Convolutional Gates', ha='center', fontsize=8, style='italic')
            y_pos -= 2.5

            # Global pooling
            ax.add_patch(FancyBboxPatch((3, y_pos-0.5), 4, 0.8,
                                        facecolor='#FFB4FF', edgecolor='black', linewidth=2))
            ax.text(5, y_pos-0.1, 'Global Avg Pool', ha='center', fontsize=10)
            y_pos -= 1.5

        # FC layer
        ax.add_patch(FancyBboxPatch((3, y_pos-0.5), 4, 0.8,
                                    facecolor='lightcoral', edgecolor='black', linewidth=2))
        ax.text(5, y_pos-0.1, 'Fully Connected', ha='center', fontsize=10)
        y_pos -= 1.5

        # Output
        ax.add_patch(FancyBboxPatch((3, y_pos-0.5), 4, 0.8,
                                    facecolor='lightgreen', edgecolor='black', linewidth=2))
        ax.text(5, y_pos-0.1, 'Output (batch, 1)', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison.png")
    plt.close()


def plot_parameter_comparison():
    """Plot parameter counts"""
    models = ['LSTMO', 'CNNLSTM', 'ConvLSTM']
    params = [199299, 210273, 148803]
    colors = ['#FFE5B4', '#B4E5FF', '#B4FFB4']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, params, color=colors, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, param in zip(bars, params):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylabel('Number of Parameters', fontsize=12, fontweight='bold')
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.title('Model Parameter Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add percentage savings
    savings = (params[0] - params[2]) / params[0] * 100
    plt.text(2, params[0] * 0.9, f'{savings:.1f}% fewer\nparameters',
             ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig('parameter_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: parameter_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("Generating ConvLSTM visualizations...")
    visualize_convlstm_cell()
    visualize_model_comparison()
    plot_parameter_comparison()
    print("\nAll visualizations generated successfully!")
