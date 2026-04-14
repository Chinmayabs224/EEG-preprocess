"""
Pipeline orchestrator for large-scale spike encoding.

Coordinates the full pipeline:
    File Discovery → Chunk Loading → Preprocessing → Spike Encoding
    → SNN Feature Extraction → Save → Visualize

Supports parallel processing via ``ProcessPoolExecutor`` and shows
progress bars via ``tqdm``.
"""

import os
import csv
import time
import numpy as np
import torch
import logging
from typing import Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .config import PipelineConfig
from .utils import FileInfo, get_file_manifest, setup_logging, Timer
from .data_loader import load_edf_chunks, load_edf_full, build_labels
from .preprocessing import preprocess_chunk
from .spike_encoder import SpikeEncoder
from .snn_feature_extractor import (
    FilterbankExtractor, SNNEncoder, extract_snn_features,
)
from .visualization import generate_all_plots


# ═════════════════════════════════════════════════════════════════
# Single-File Processor
# ═════════════════════════════════════════════════════════════════

def process_single_file(
    file_info: FileInfo,
    config: PipelineConfig,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Process one EDF file through the full pipeline.

    Steps:
        1. Load EDF in chunks
        2. Preprocess each chunk
        3. Concatenate all chunks
        4. Encode spikes with selected method
        5. Extract SNN features (filterbank → SNN encoder)
        6. Save spike encoding as ``.npz``

    Parameters
    ----------
    file_info : FileInfo
        Named tuple with file path, subject ID, seizure intervals, etc.
    config : PipelineConfig
    logger : logging.Logger, optional

    Returns
    -------
    dict with processing statistics.
    """
    start_time = time.perf_counter()

    fname = file_info.filename
    subject_id = file_info.subject_id

    if logger:
        logger.info(f"  Processing: {subject_id}/{fname}")

    # ── Step 1 + 2: Load and preprocess in chunks ────────────────
    all_chunks = []
    fs = config.sampling_rate  # will be overwritten from file

    try:
        for chunk, chunk_start, chunk_fs in load_edf_chunks(
            file_info.path, config, logger
        ):
            fs = chunk_fs
            cleaned = preprocess_chunk(chunk, fs, config)
            all_chunks.append(cleaned)
    except Exception as exc:
        if logger:
            logger.error(f"  ✗ Failed to load {fname}: {exc}")
        return {
            "filename": fname,
            "subject_id": subject_id,
            "status": "FAILED",
            "error": str(exc),
        }

    if not all_chunks:
        if logger:
            logger.warning(f"  ⚠ No data loaded from {fname}")
        return {
            "filename": fname,
            "subject_id": subject_id,
            "status": "EMPTY",
        }

    # Concatenate all chunks into full preprocessed recording
    preprocessed = np.concatenate(all_chunks, axis=1)
    n_ch, n_samples = preprocessed.shape
    duration_sec = n_samples / fs

    if logger:
        logger.debug(
            f"    Preprocessed: {n_ch} ch × {n_samples} samples "
            f"({duration_sec:.1f}s)"
        )

    # ── Step 3: Build labels ─────────────────────────────────────
    labels = build_labels(
        n_samples, file_info.seizure_intervals, fs
    )

    # ── Step 4: Spike encoding ───────────────────────────────────
    encoder = SpikeEncoder(config)
    encode_result = encoder.encode(preprocessed, fs)
    spikes = encode_result["spikes"]
    spike_metadata = encode_result["metadata"]

    if logger:
        logger.debug(
            f"    Spikes: {spike_metadata['total_spikes']} total, "
            f"mean rate={spike_metadata['mean_firing_rate_hz']:.1f} Hz"
        )

    # ── Step 5: SNN feature extraction ───────────────────────────
    snn_result = None
    try:
        snn_result = extract_snn_features(
            preprocessed, labels["binary"], config, logger=logger,
        )
    except Exception as exc:
        if logger:
            logger.warning(
                f"    ⚠ SNN feature extraction failed for {fname}: {exc}. "
                f"Saving spikes only."
            )

    # ── Step 6: Save ─────────────────────────────────────────────
    save_dir = os.path.join(config.spikes_dir, subject_id)
    os.makedirs(save_dir, exist_ok=True)

    base_name = os.path.splitext(fname)[0]
    save_path = os.path.join(save_dir, f"{base_name}.npz")

    save_dict = {
        "spikes": spikes,
        "binary_labels": labels["binary"],
        "preictal_labels": labels["preictal"],
        "seizure_intervals": np.array(
            file_info.seizure_intervals
            if file_info.seizure_intervals else [[-1, -1]],
            dtype=np.int32,
        ),
        "spike_method": config.spike_method,
        "sampling_rate": fs,
        "n_channels": n_ch,
    }

    if snn_result is not None:
        save_dict["snn_features"] = snn_result["snn_features"]
        save_dict["filterbank_features"] = snn_result["filterbank_features"]
        save_dict["snn_labels"] = snn_result["labels"]

    np.savez_compressed(save_path, **save_dict)

    elapsed = time.perf_counter() - start_time
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)

    if logger:
        logger.info(
            f"  ✓ {subject_id}/{fname} → {base_name}.npz "
            f"({file_size_mb:.1f} MB) in {elapsed:.1f}s"
        )

    return {
        "filename": fname,
        "subject_id": subject_id,
        "status": "OK",
        "n_channels": n_ch,
        "n_samples": n_samples,
        "duration_sec": round(duration_sec, 1),
        "has_seizure": file_info.has_seizure,
        "n_seizures": len(file_info.seizure_intervals),
        "spike_method": config.spike_method,
        "total_spikes": spike_metadata["total_spikes"],
        "mean_firing_rate_hz": round(
            spike_metadata["mean_firing_rate_hz"], 2
        ),
        "snn_features_shape": (
            str(snn_result["snn_features"].shape) if snn_result else "N/A"
        ),
        "output_path": save_path,
        "output_size_mb": round(file_size_mb, 2),
        "processing_time_sec": round(elapsed, 2),
    }


# ═════════════════════════════════════════════════════════════════
# Worker wrapper for multiprocessing
# ═════════════════════════════════════════════════════════════════

def _worker(args):
    """Wrapper for process_single_file for use with ProcessPoolExecutor."""
    file_info, config = args
    # Each worker creates its own minimal logger (file handler only)
    return process_single_file(file_info, config, logger=None)


# ═════════════════════════════════════════════════════════════════
# Full Pipeline Orchestrator
# ═════════════════════════════════════════════════════════════════

def run_full_pipeline(config: PipelineConfig):
    """
    Run the complete spike encoding pipeline on all EDF files.

    Steps:
        1. Create output directories
        2. Set up logging
        3. Discover all EDF files (recursive)
        4. Process files in parallel
        5. Generate verification plots for a sample
        6. Write summary CSV

    Parameters
    ----------
    config : PipelineConfig
    """
    # ── Setup ────────────────────────────────────────────────────
    config.create_dirs()
    logger = setup_logging(config.logs_dir)

    logger.info("=" * 60)
    logger.info("  SPIKE ENCODING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  Data dir     : {config.data_dir}")
    logger.info(f"  Output dir   : {config.output_dir}")
    logger.info(f"  Spike method : {config.spike_method}")
    logger.info(f"  Chunk size   : {config.chunk_duration_sec}s")
    logger.info(f"  Workers      : {config.n_workers}")
    logger.info(f"  Device       : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("=" * 60)

    # ── Step 1: Discover files ───────────────────────────────────
    with Timer("File discovery", logger):
        manifest = get_file_manifest(config.data_dir, logger)

    if not manifest:
        logger.error("No EDF files found. Exiting.")
        return

    # ── Step 2: Process all files ────────────────────────────────
    logger.info(f"\n{'─' * 60}")
    logger.info(f"  Processing {len(manifest)} EDF files ...")
    logger.info(f"{'─' * 60}")

    results = []

    if config.n_workers <= 1:
        # Sequential processing (easier debugging)
        iterator = manifest
        if HAS_TQDM:
            iterator = tqdm(manifest, desc="Processing files",
                           unit="file")
        for file_info in iterator:
            result = process_single_file(file_info, config, logger)
            results.append(result)
    else:
        # Parallel processing
        work_items = [(fi, config) for fi in manifest]

        with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
            futures = {
                executor.submit(_worker, item): item[0].filename
                for item in work_items
            }

            if HAS_TQDM:
                pbar = tqdm(total=len(futures), desc="Processing files",
                           unit="file")

            for future in as_completed(futures):
                fname = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if logger and result.get("status") == "OK":
                        logger.info(
                            f"  ✓ {result['subject_id']}/{fname} "
                            f"({result.get('processing_time_sec', 0):.1f}s)"
                        )
                except Exception as exc:
                    logger.error(f"  ✗ {fname}: {exc}")
                    results.append({
                        "filename": fname,
                        "status": "FAILED",
                        "error": str(exc),
                    })

                if HAS_TQDM:
                    pbar.update(1)

            if HAS_TQDM:
                pbar.close()

    # ── Step 3: Generate verification plots ──────────────────────
    logger.info(f"\n{'─' * 60}")
    logger.info(f"  Generating verification plots ...")
    logger.info(f"{'─' * 60}")

    _generate_sample_plots(manifest, config, logger)

    # ── Step 4: Write summary CSV ────────────────────────────────
    _write_summary_csv(results, config, logger)

    # ── Final report ─────────────────────────────────────────────
    n_ok = sum(1 for r in results if r.get("status") == "OK")
    n_fail = sum(1 for r in results if r.get("status") == "FAILED")
    n_seizure = sum(1 for r in results
                    if r.get("status") == "OK" and r.get("has_seizure"))
    total_spikes = sum(r.get("total_spikes", 0) for r in results
                       if r.get("status") == "OK")

    logger.info(f"\n{'═' * 60}")
    logger.info(f"  PIPELINE COMPLETE")
    logger.info(f"{'═' * 60}")
    logger.info(f"  Files processed  : {n_ok}/{len(manifest)}")
    logger.info(f"  Files failed     : {n_fail}")
    logger.info(f"  Seizure files    : {n_seizure}")
    logger.info(f"  Total spikes     : {total_spikes:,}")
    logger.info(f"  Encoding method  : {config.spike_method}")
    logger.info(f"  Output directory : {config.output_dir}")
    logger.info(f"{'═' * 60}")


# ═════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════

def _generate_sample_plots(
    manifest: list,
    config: PipelineConfig,
    logger: logging.Logger,
):
    """Generate verification plots for a sample of files."""
    # Prefer files with seizures for more interesting plots
    seizure_files = [f for f in manifest if f.has_seizure]
    non_seizure = [f for f in manifest if not f.has_seizure]

    # Pick a mix: prefer seizure files, fill with non-seizure
    sample = seizure_files[:config.plot_samples]
    remaining = config.plot_samples - len(sample)
    if remaining > 0 and non_seizure:
        sample.extend(non_seizure[:remaining])

    if not sample:
        logger.warning("  No files available for plot generation")
        return

    encoder = SpikeEncoder(config)

    for file_info in sample:
        file_id = os.path.splitext(file_info.filename)[0]
        logger.info(f"  Plotting: {file_info.subject_id}/{file_info.filename}")

        try:
            # Load full file for visualization
            signal, fs = load_edf_full(file_info.path, config, logger)
            cleaned = preprocess_chunk(signal, fs, config)

            # Encode with selected method
            result = encoder.encode(cleaned, fs)
            spikes = result["spikes"]

            # Also encode with all methods for the comparison plot
            all_method_spikes = encoder.encode_all_methods(cleaned, fs)
            all_spikes_dict = {
                m: r["spikes"] for m, r in all_method_spikes.items()
            }

            # Generate all plots
            generate_all_plots(
                signal=cleaned,
                spikes=spikes,
                fs=fs,
                config=config,
                file_id=file_id,
                seizure_intervals=(
                    file_info.seizure_intervals if file_info.has_seizure
                    else None
                ),
                all_method_spikes=all_spikes_dict,
                logger=logger,
            )

        except Exception as exc:
            logger.warning(
                f"  ⚠ Plot generation failed for {file_info.filename}: {exc}"
            )


def _write_summary_csv(
    results: list,
    config: PipelineConfig,
    logger: logging.Logger,
):
    """Write processing summary as CSV."""
    csv_path = os.path.join(config.chunks_dir, "processing_summary.csv")

    # Determine columns from first successful result
    ok_results = [r for r in results if r.get("status") == "OK"]
    if not ok_results:
        logger.warning("  No successful results to write to CSV")
        return

    fieldnames = list(ok_results[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    logger.info(f"  📋 Summary CSV: {csv_path}")
