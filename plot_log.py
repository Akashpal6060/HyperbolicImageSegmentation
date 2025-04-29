#!/usr/bin/env python3
import re
import os
import argparse
import matplotlib.pyplot as plt

def parse_log(log_path):
    """
    Reads the log file and extracts:
      - epoch number
      - training loss
      - validation accuracy
    Returns three lists: epochs, losses, val_accs.
    """
    epoch_nums = []
    losses     = []
    val_accs   = []
    # Matches lines like:
    # INFO:__main__:Epoch 1/500 — loss: 0.3019, val acc: 0.7488
    pattern = re.compile(
        r'Epoch\s+(\d+)/\d+\s+—\s+loss:\s*([0-9]+(?:\.[0-9]+)?),\s*val acc:\s*([0-9]+(?:\.[0-9]+)?)'
    )
    with open(log_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch_nums.append(int(m.group(1)))
                losses.    append(float(m.group(2)))
                val_accs.  append(float(m.group(3)))
    return epoch_nums, losses, val_accs

def main():
    parser = argparse.ArgumentParser(
        description="Plot training loss & validation accuracy from a training log."
    )
    parser.add_argument(
        '--log_file', 
        required=True, 
        help='Path to train_IDDAW_gpu2_bs64.log'
    )
    parser.add_argument(
        '--output', 
        default='/users/student/pg/pg23/akash.pal/HyperbolicImageSegmentation/logs/training_curves.png', 
        help='Where to save the plot (PNG)'
    )
    args = parser.parse_args()

    # ensure output directory exists
    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)

    epochs, losses, accs = parse_log(args.log_file)
    if not epochs:
        raise RuntimeError("No epoch lines found in log. Check the regex or log format.")

    plt.figure()
    plt.plot(epochs, losses,   label='Train Loss')
    plt.plot(epochs, accs,     label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss & Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"✅ Saved plot to {args.output}")

if __name__ == '__main__':
    main()
