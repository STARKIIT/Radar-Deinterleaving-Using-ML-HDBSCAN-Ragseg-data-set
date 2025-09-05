import h5py
import argparse
import numpy as np
import os

def print_h5_structure(file_path):
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        print(f"Keys in {file_path}: {keys}")
        for key in keys:
            dset = f[key]
            print(f"  - {key}: shape={dset.shape}, dtype={dset.dtype}")
        print(f"Total number of keys: {len(keys)}")
        return keys

def convert_h5_to_csv(file_path, key, out_csv):
    with h5py.File(file_path, 'r') as f:
        if key not in f:
            print(f"Key '{key}' not found in file.")
            return
        data = f[key][:]
        np.savetxt(out_csv, data, delimiter=",")
        print(f"Saved dataset '{key}' to CSV: {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Inspect HDF5 file and optionally convert a dataset to CSV.")
    parser.add_argument("file_path", help="Path to the HDF5 file to inspect.")
    parser.add_argument("--csv", help="Convert specified key to CSV (provide key name).")
    parser.add_argument("--out", help="Output CSV file path (required if --csv is used).")
    args = parser.parse_args()

    keys = print_h5_structure(args.file_path)

    if args.csv:
        if not args.out:
            print("Please provide --out <output_csv_path> when using --csv.")
        else:
            convert_h5_to_csv(args.file_path, args.csv, args.out)

if __name__ == "__main__":
    main()
