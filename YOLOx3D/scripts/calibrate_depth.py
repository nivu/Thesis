import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.depth_calibration import calibrate_with_kitti_3D_objects_dataset, reset_calibration

def plot_calibration_results(rel_depths, abs_depths, coeffs, save_path=None):
    """Plot calibration results with fit curve and residuals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    rel_depths = np.array(rel_depths)
    abs_depths = np.array(abs_depths)

    # Scatter + fit curve
    ax1.scatter(rel_depths, abs_depths, alpha=0.5, s=1)
    x_fit = np.linspace(rel_depths.min(), rel_depths.max(), 100)
    y_fit = coeffs[0]*x_fit**2 + coeffs[1]*x_fit + coeffs[2]
    ax1.plot(x_fit, y_fit, 'r-', linewidth=2, label='Fitted curve')
    ax1.set(xlabel='Relative Depth', ylabel='Absolute Depth (m)', title='Depth Calibration')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Residuals
    pred_abs = coeffs[0]*rel_depths**2 + coeffs[1]*rel_depths + coeffs[2]
    residuals = abs_depths - pred_abs
    ax2.scatter(rel_depths, residuals, alpha=0.5, s=1)
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set(xlabel='Relative Depth', ylabel='Residuals (m)', title='Calibration Residuals')
    ax2.grid(alpha=0.3)

    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    stats_text = f'RMSE: {rmse:.3f}m\nMAE: {mae:.3f}m\nPoints: {len(rel_depths)}'
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration plot saved to: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Calibrate depth estimation with KITTI dataset')
    parser.add_argument('kitti_path', help='Path to KITTI dataset')
    parser.add_argument('--max-images', type=int, default=200, help='Max images to process')
    parser.add_argument('--max-depth', type=float, default=100.0, help='Max depth (m) to consider')
    parser.add_argument('--depth-model-size', choices=['small', 'base', 'large'], default='small', help='Depth model size')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto', help='Computation device')
    parser.add_argument('--output', default='calibration/depth_calibration.json', help='Output calibration file')
    args = parser.parse_args()

    print("="*60)
    print("KITTI DEPTH CALIBRATION")
    print("="*60)

    if not os.path.exists(args.kitti_path):
        print(f"Error: KITTI path not found: {args.kitti_path}")
        return 1

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"KITTI dataset: {args.kitti_path}")
    print(f"Max images: {args.max_images}")
    print(f"Max depth: {args.max_depth} m")
    print(f"Depth model size: {args.depth_model_size}")
    print(f"Device: {args.device}")
    print(f"Output file: {output_path}")

    reset_calibration()

    print("\n" + "="*60)
    print("STARTING CALIBRATION")
    print("="*60)

    try:
        coeffs, num_points, rel_depths, abs_depths = calibrate_with_kitti_3D_objects_dataset(
            kitti_base_dir=args.kitti_path,
            depth_model_size=args.depth_model_size,
            device=args.device,
            max_depth=args.max_depth,
            max_images=args.max_images,
            save_file=str(output_path),
            show_progress=True,
            return_data=True
        )

        print("\n" + "="*60)
        print("CALIBRATION COMPLETED")
        print("="*60)
        print(f"Calibration coefficients: {coeffs}")
        print(f"Number of data points: {num_points}")
        print(f"Calibration saved to: {output_path}")

        plot_path = output_path.parent / 'calibration_plot.png'
        plot_calibration_results(rel_depths, abs_depths, coeffs, save_path=plot_path)

        return 0

    except Exception as e:
        print(f"Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()