#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path

def split_sft_data(input_file, train_file, val_file, val_ratio=0.1, seed=42):
    """Split SFT JSONL data into training and validation sets."""
    
    random.seed(seed)
    
    with open(input_file, 'r') as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]
    
    # random.shuffle(data)
    
    val_size = int(len(data) * val_ratio)
    train_size = len(data) - val_size
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    Path(train_file).parent.mkdir(parents=True, exist_ok=True)
    Path(val_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(val_file, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Split complete:")
    print(f"  Total samples: {len(data)}")
    print(f"  Training samples: {len(train_data)} -> {train_file}")
    print(f"  Validation samples: {len(val_data)} -> {val_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split SFT JSONL data into train/val sets")
    parser.add_argument("--input", default="MATH/sft.jsonl", help="Input JSONL file")
    parser.add_argument("--train-output", default="MATH/sft_train.jsonl", help="Training output file")
    parser.add_argument("--val-output", default="MATH/sft_val.jsonl", help="Validation output file")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    split_sft_data(
        input_file=args.input,
        train_file=args.train_output,
        val_file=args.val_output,
        val_ratio=args.val_ratio,
        seed=args.seed
    )