import os
import torch
import json
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Build a balanced test set.")
    parser.add_argument('--data_dir', type=str, default='/mnt/waymo_data/waymo_processed_samples/testing',
                        help='Directory containing .pt test files')
    parser.add_argument('--output_json', type=str, default='balanced_test_files.json',
                        help='Path to save selected test file list')
    parser.add_argument('--total_samples', type=int, default=2000,
                        help='Total number of test files to select')
    parser.add_argument('--ratios', type=int, nargs='+', default=[1, 1, 1],
                        help='Class ratio as 3 integers for [vehicle, pedestrian, cyclist]')
    return parser.parse_args()

def main():
    args = parse_args()

    class_map = {
        1: 'vehicle',
        2: 'pedestrian',
        4: 'cyclist'
    }

    desired_ratios = {
        1: args.ratios[0],
        2: args.ratios[1],
        4: args.ratios[2]
    }

    total_ratio = sum(desired_ratios.values())
    quota_per_class = {
        cls: int(args.total_samples * (ratio / total_ratio))
        for cls, ratio in desired_ratios.items()
    }

    print(f"Target per-class file quotas: {quota_per_class}")

    selected = {1: [], 2: [], 4: []}
    used_files = set()

    all_files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.pt')])

    for fname in all_files:
        if all(len(selected[c]) >= quota_per_class[c] for c in quota_per_class):
            break  # All quotas filled

        fpath = os.path.join(args.data_dir, fname)

        try:
            sample = torch.load(fpath)
            labels = sample.get('labels', [])

            if not isinstance(labels, torch.Tensor):
                continue

            unique_labels = set(labels.tolist())

            for cls in desired_ratios:
                if cls in unique_labels and len(selected[cls]) < quota_per_class[cls] and fname not in used_files:
                    selected[cls].append(fname)
                    used_files.add(fname)
                    break  # Avoid double counting

        except Exception as e:
            print(f"⚠️ Skipping {fname}: {e}")

    # Combine and save
    all_selected = sorted(set(fname for lst in selected.values() for fname in lst))
    print(f"\n✅ Total selected files: {len(all_selected)}")
    for cls, lst in selected.items():
        print(f"  - {class_map[cls]} ({cls}): {len(lst)}")

    with open(args.output_json, 'w') as f:
        json.dump(all_selected, f, indent=2)
    print(f"\n📁 Saved to: {args.output_json}")

if __name__ == "__main__":
    main()
