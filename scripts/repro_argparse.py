import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, nargs='?', const="", default="default_dir")
    args = parser.parse_args()
    print(f"Accepted args: {args}")
    print(f"dataset_dir value: '{args.dataset_dir}'")

if __name__ == "__main__":
    print(f"Raw sys.argv: {sys.argv}")
    parse_args()
