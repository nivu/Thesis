"""
Check Wheel Segmentation Training Status
"""

from pathlib import Path
import pandas as pd
from datetime import datetime


def check_status():
    print("=" * 70)
    print("Wheel Segmentation Training Status")
    print("=" * 70)

    # Check if training is running
    weights_dir = Path("runs/segment/wheel_seg/weights")
    results_file = Path("runs/segment/wheel_seg/results.csv")

    if not weights_dir.exists():
        print("\n⏳ Training has not started yet")
        print("   Waiting for training process to initialize...")
        return

    # Check model files
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"

    if best_pt.exists():
        size_mb = best_pt.stat().st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(best_pt.stat().st_mtime)
        print(f"\n✓ Model file: {best_pt}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Last updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check training results
    if results_file.exists():
        try:
            df = pd.read_csv(results_file)
            current_epoch = len(df)
            total_epochs = 50

            print(f"\n📊 Training Progress:")
            print(f"  Current Epoch: {current_epoch}/{total_epochs}")
            print(f"  Progress: {(current_epoch/total_epochs)*100:.1f}%")

            if current_epoch > 0:
                latest = df.iloc[-1]
                print(f"\n  Latest Metrics (Epoch {current_epoch}):")
                print(f"    Box Loss: {latest['train/box_loss']:.3f}")
                print(f"    Seg Loss: {latest['train/seg_loss']:.3f}")
                print(f"    Cls Loss: {latest['train/cls_loss']:.3f}")

                if 'metrics/mAP50(M)' in df.columns:
                    print(f"    mAP@50: {latest['metrics/mAP50(M)']:.3f}")

            # Estimate remaining time
            if current_epoch > 1:
                # Rough estimate: ~3.3 minutes per epoch
                remaining_epochs = total_epochs - current_epoch
                remaining_time_min = remaining_epochs * 3.3
                remaining_hours = remaining_time_min / 60
                print(f"\n⏱  Estimated time remaining: {remaining_hours:.1f} hours")

        except Exception as e:
            print(f"\n⚠️  Could not parse results.csv: {e}")
    else:
        print("\n⏳ Training started but no results yet")
        print("   Check back in a few minutes...")

    print("\n" + "=" * 70)
    print("TIP: Run this script periodically to check progress")
    print("     or check: runs/segment/wheel_seg/results.csv")
    print("=" * 70)


if __name__ == "__main__":
    check_status()
