from datasets import load_dataset
import json
import os

# Available configs in the MATH dataset
configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

# First, let's explore the structure of one config
print("Exploring dataset structure...")
sample_ds = load_dataset("EleutherAI/hendrycks_math", configs[0])
print(f"Available splits: {list(sample_ds.keys())}")
if 'train' in sample_ds:
    print(f"Train size: {len(sample_ds['train'])}")
    print(f"Sample train item: {sample_ds['train'][0]}")
if 'test' in sample_ds:
    print(f"Test size: {len(sample_ds['test'])}")
    print(f"Sample test item: {sample_ds['test'][0]}")

# Collect all data from all configs
all_train_data = []
all_test_data = []

print("\nDownloading all configs...")
for config in configs:
    print(f"Processing {config}...")
    ds = load_dataset("EleutherAI/hendrycks_math", config)
    
    if 'train' in ds:
        for item in ds['train']:
            # Add config information to each item
            item_with_config = dict(item)
            item_with_config['config'] = config
            all_train_data.append(item_with_config)
    
    if 'test' in ds:
        for item in ds['test']:
            # Add config information to each item
            item_with_config = dict(item)
            item_with_config['config'] = config
            all_test_data.append(item_with_config)

print(f"\nTotal train samples: {len(all_train_data)}")
print(f"Total test samples: {len(all_test_data)}")

# Save to JSONL files
print("\nSaving to JSONL files...")
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in all_train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

with open('test.jsonl', 'w', encoding='utf-8') as f:
    for item in all_test_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("✅ Successfully saved train.jsonl and test.jsonl files!")
