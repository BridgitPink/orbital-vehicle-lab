from pathlib import Path
from collections import Counter

TRAIN_LIST = Path("C:/Users/bridg/orbital-vehicle-lab/data/raw/cowc_detection/COWC_train_list_detection.txt")
TEST_LIST = Path("C:/Users/bridg/orbital-vehicle-lab/data/raw/cowc_detection/COWC_test_list_detection.txt")


def inspect_list(path: Path, max_lines: int = 15):
    print(f"\nInspecting: {path}")
    if not path.exists():
        print("File not found.")
        return

    labels = Counter()
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    print(f"Total lines: {len(lines)}")
    print("\nSample lines:")
    for line in lines[:max_lines]:
        print(line)

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            labels[parts[-1]] += 1

    print("\nLabel counts:")
    for k, v in sorted(labels.items()):
        print(f"  {k}: {v}")


def main():
    inspect_list(TRAIN_LIST)
    inspect_list(TEST_LIST)


if __name__ == "__main__":
    main()