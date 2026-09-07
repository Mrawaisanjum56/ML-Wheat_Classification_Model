
import os
import shutil
import random
from pathlib import Path

RAW_DATA_DIR    = "raw_data"
OUTPUT_DIR      = "dataset"
CLASS_NAMES     = ["good", "average", "bad"]
SPLIT_RATIO     = (0.70, 0.15, 0.15)  
RANDOM_SEED     = 42
MIN_PER_CLASS   = 80
TARGET_TRAIN    = 200

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def _list_images(folder: Path):
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def check_minimum_samples():
    print(" Image Count Check")
    counts = {}
    for cls in CLASS_NAMES:
        src = Path(RAW_DATA_DIR) / cls
        if not src.exists():
            print(f"  {cls:8s}: WARNING MISSING create folder raw_data/{cls}/")
            counts[cls] = 0
            continue
        counts[cls] = len(_list_images(src))

    max_count = max(counts.values()) if counts else 1
    for cls, n in counts.items():
        bar = '█' * int(n / max(max_count, 1) * 30)
        status = "OK" if n >= MIN_PER_CLASS else f"LOW — need {MIN_PER_CLASS}+ images"
        print(f"  {cls:8s}: {n:4d}  {bar:<30}  {status}")
    print("─" * 52)

    vals = [v for v in counts.values() if v > 0]
    if len(vals) > 1:
        ratio = max(vals) / min(vals)
        minority = min(counts, key=lambda k: counts[k] if counts[k] > 0 else 9999)
        if ratio > 3:
            print(f"\n  CLASS IMBALANCE DETECTED (ratio {ratio:.1f}:1)")
            print(f"  '{minority}' class has far fewer images.")
            print(f"  Oversampling will be applied during dataset split.")
            print(f"  Strongly recommended: collect more '{minority}' images.\n")
        else:
            print(f" Class balance looks reasonable (ratio {ratio:.1f}:1)")

    return counts


def split_dataset(counts):
    random.seed(RANDOM_SEED)
    splits = ['train', 'val', 'test']
    stats = {s: {c: 0 for c in CLASS_NAMES} for s in splits}

    for cls in CLASS_NAMES:
        src = Path(RAW_DATA_DIR) / cls
        if not src.exists() or counts[cls] == 0:
            print(f"Skipping '{cls}' — no images found.")
            continue

        images = _list_images(src)
        random.shuffle(images)

        n = len(images)
        n_tr = int(n * SPLIT_RATIO[0])
        n_val = int(n * SPLIT_RATIO[1])

        n_val = max(n_val, 1)
        n_te = max(n - n_tr - n_val, 1)
        n_tr = n - n_val - n_te

        buckets = {
            'train': images[:n_tr],
            'val':   images[n_tr:n_tr + n_val],
            'test':  images[n_tr + n_val:]
        }

        for split, files in buckets.items():
            dst = Path(OUTPUT_DIR) / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dst / f.name)
            stats[split][cls] = len(files)

        
        train_dst = Path(OUTPUT_DIR) / 'train' / cls
        current_n = len(buckets['train'])
        target = max(TARGET_TRAIN, current_n)

        if current_n < target:
            all_train = list(train_dst.iterdir())
            added = 0
            while current_n + added < target:
                src_file = random.choice(all_train)
                new_name = f"{src_file.stem}_dup{added:04d}{src_file.suffix}"
                shutil.copy2(src_file, train_dst / new_name)
                added += 1
            stats['train'][cls] = current_n + added
            print(f"[{cls:8s}] original={current_n}  oversampled to "
                  f"{current_n + added}  val={len(buckets['val'])}  "
                  f"test={len(buckets['test'])}")
        else:
            print(f"[{cls:8s}] train={current_n}  val={len(buckets['val'])}  "
                  f"test={len(buckets['test'])}")

    print("\nDataset Split Summary")
    for split in splits:
        total = sum(stats[split].values())
        row = "  ".join(f"{c}={stats[split][c]}" for c in CLASS_NAMES)
        print(f"  {split:6s}: {total:4d} images    {row}")
    print("─" * 52)
    print(f"Dataset saved → {OUTPUT_DIR}/\n")

    _save_stats(stats)
    return stats


def _save_stats(stats):
    lines = ["Wheat Dataset Split Statistics", "=" * 40]
    for split in ['train', 'val', 'test']:
        total = sum(stats[split].values())
        lines.append(f"\n{split.upper()} ({total} total):")
        for cls in CLASS_NAMES:
            lines.append(f"  {cls:8s}: {stats[split][cls]}")
    lines.append("\nNote: 'train' may include oversampled duplicates.")
    with open("dataset_stats.txt", "w") as f:
        f.write("\n".join(lines))
    print("Dataset stats saved → dataset_stats.txt")


def plot_dataset_distribution(stats):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    colors = ['limegreen', 'orange', 'red']

    for ax, split in zip(axes, ['train', 'val', 'test']):
        values = [stats[split][c] for c in CLASS_NAMES]
        bars = ax.bar(CLASS_NAMES, values, color=colors, edgecolor='white', width=0.5)
        ax.set_title(f"{split.capitalize()} Set", fontsize=12, fontweight='bold')
        ax.set_ylabel("Image Count")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', fontsize=10)
        max_val = max(max(stats[s].values()) for s in ['train', 'val', 'test'])
        ax.set_ylim(0, max_val * 1.15)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Wheat Dataset — Class Distribution per Split',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=150)
    plt.close()
    print("Distribution chart saved → dataset_distribution.png")


def verify_output():
    print("\n Output Verification")
    from PIL import Image as PILImage

    errors = 0
    for split in ['train', 'val', 'test']:
        for cls in CLASS_NAMES:
            folder = Path(OUTPUT_DIR) / split / cls
            if not folder.exists():
                print(f"  MISSING: {folder}")
                errors += 1
                continue
            files = list(folder.iterdir())
            bad = []
            for f in files:
                try:
                    PILImage.open(f).verify()
                except Exception:
                    bad.append(f.name)
            if bad:
                print(f"  {split}/{cls}: {len(bad)} corrupt file(s): {bad[:3]}")
                errors += 1
            else:
                print(f"  OK  {split}/{cls}: {len(files)} images")

    if errors == 0:
        print("\n  All files verified — dataset is clean.\n")
    else:
        print(f"\n  {errors} issue(s) found.\n")
    print("─" * 52)


if __name__ == "__main__":
    print("=" * 52)
    print("  Wheat Dataset Preparation Tool")
    print("=" * 52)

    counts = check_minimum_samples()
    if sum(counts.values()) == 0:
        print("\nERROR: No images found in raw_data/. Please add images before running.")
        exit(1)

    stats = split_dataset(counts)
    plot_dataset_distribution(stats)
    verify_output()

    print("Done!")
   