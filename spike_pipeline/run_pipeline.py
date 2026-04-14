#!/usr/bin/env python3
"""
CLI entry point for the large-scale EEG spike encoding pipeline.

Usage examples::

    # Default: LIF encoding, 4 workers, data at /home/ubuntu/DESTINATION
    python -m spike_pipeline.run_pipeline

    # Rate coding, 8 workers
    python -m spike_pipeline.run_pipeline --spike_method rate --n_workers 8

    # Temporal coding, custom dirs
    python -m spike_pipeline.run_pipeline \\
        --data_dir /home/ubuntu/DESTINATION \\
        --output_dir ./output \\
        --spike_method temporal \\
        --n_workers 4 \\
        --chunk_duration 60 \\
        --plot_samples 5

    # Sequential processing (for debugging)
    python -m spike_pipeline.run_pipeline --n_workers 1
"""

import argparse
import sys

from .config import PipelineConfig
from .pipeline import run_full_pipeline


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Large-Scale EEG Spike Encoding Pipeline — "
            "Processes CHB-MIT dataset and generates spike encodings + "
            "SNN features for all EDF files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages (matching architecture diagram):
  1. Raw EEG Input      → recursive EDF discovery
  2. Preprocessing      → bandpass, notch, CAR, z-score
  3. Windowing+Labeling → 2-sec epochs + seizure annotations
  4. Spike Encoding     → rate / LIF / temporal coding
  5. SNN Feature Ext.   → LIF layers → spike trains (T×64)
  6. Save + Visualize   → .npz files + 7 plot types

Example on EC2:
  python -m spike_pipeline.run_pipeline \\
      --data_dir /home/ubuntu/DESTINATION \\
      --spike_method lif --n_workers 8
""",
    )

    # Paths
    parser.add_argument(
        "--data_dir", type=str, default="/home/ubuntu/DESTINATION",
        help="Root directory containing chbXX/ folders (default: /home/ubuntu/DESTINATION)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output",
        help="Output directory for spikes, plots, logs (default: ./output)",
    )

    # Spike encoding
    parser.add_argument(
        "--spike_method", type=str, default="lif",
        choices=["rate", "lif", "temporal"],
        help="Spike encoding method (default: lif)",
    )

    # Scalability
    parser.add_argument(
        "--n_workers", type=int, default=4,
        help="Number of parallel worker processes (default: 4, use 1 for debugging)",
    )
    parser.add_argument(
        "--chunk_duration", type=int, default=60,
        help="Duration of each chunk in seconds for memory-safe loading (default: 60)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size for SNN inference (default: 256)",
    )

    # Visualization
    parser.add_argument(
        "--plot_samples", type=int, default=5,
        help="Number of sample files to generate verification plots for (default: 5)",
    )
    parser.add_argument(
        "--plot_duration", type=float, default=10.0,
        help="Duration in seconds to show in signal plots (default: 10.0)",
    )

    # Signal parameters
    parser.add_argument(
        "--n_channels", type=int, default=23,
        help="Max number of EEG channels to read (default: 23)",
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=256,
        help="Expected sampling rate in Hz (default: 256)",
    )

    # LIF parameters
    parser.add_argument(
        "--lif_threshold", type=float, default=1.5,
        help="LIF encoding threshold in std devs (default: 1.5)",
    )
    parser.add_argument(
        "--lif_beta", type=float, default=0.85,
        help="LIF membrane decay factor (default: 0.85)",
    )

    # Rate coding parameters
    parser.add_argument(
        "--rate_max_hz", type=float, default=100.0,
        help="Max firing rate for rate coding in Hz (default: 100)",
    )

    # Temporal coding parameters
    parser.add_argument(
        "--temporal_n_bins", type=int, default=8,
        help="Number of time bins for temporal coding (default: 8)",
    )

    # Random seed
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    # Build config from CLI arguments
    config = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        spike_method=args.spike_method,
        n_workers=args.n_workers,
        chunk_duration_sec=args.chunk_duration,
        batch_size=args.batch_size,
        plot_samples=args.plot_samples,
        plot_duration_sec=args.plot_duration,
        n_channels=args.n_channels,
        sampling_rate=args.sampling_rate,
        lif_threshold=args.lif_threshold,
        lif_beta=args.lif_beta,
        rate_max_hz=args.rate_max_hz,
        temporal_n_bins=args.temporal_n_bins,
        random_seed=args.seed,
    )

    print()
    print("█" * 60)
    print("█  LARGE-SCALE EEG SPIKE ENCODING PIPELINE")
    print("█" + "═" * 58 + "█")
    print(f"█  Data dir     : {config.data_dir}")
    print(f"█  Output dir   : {config.output_dir}")
    print(f"█  Spike method : {config.spike_method}")
    print(f"█  Workers      : {config.n_workers}")
    print(f"█  Chunk size   : {config.chunk_duration_sec}s")
    print("█" * 60)
    print()

    try:
        run_full_pipeline(config)
    except KeyboardInterrupt:
        print("\n⚠ Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n✗ Pipeline failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
