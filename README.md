# Complete Multi-Task EEG Analysis Pipeline

A modular EEG analysis pipeline for seizure prediction, disease classification, interpretable diagnosis, and anomaly detection. The project is organized around a shared preprocessing and SNN feature-extraction stage, then branches into four downstream models:

- LSTM + Transformer for seizure prediction
- CNN for disease classification
- Random Forest for interpretable diagnosis
- Autoencoder for anomaly detection

## Project Structure

```text
eeg_pipeline/
├── config.py
├── data_loader.py
├── preprocessor.py
├── snn_encoder.py
├── models/
│   ├── lstm_transformer.py
│   ├── cnn_classifier.py
│   ├── random_forest.py
│   └── autoencoder.py
├── trainer.py
├── evaluator.py
├── visualizer.py
└── main.py
```

## What the Pipeline Does

The pipeline is designed to:

- Load EDF EEG recordings
- Apply preprocessing such as artifact clipping, notch filtering, bandpass filtering, common average referencing, and normalization
- Extract filterbank features and encode them with a spiking neural network
- Train multiple downstream models for different EEG tasks
- Evaluate and visualize model performance

## Requirements

Install the dependencies listed in [requirements.txt](requirements.txt).

On Windows, it is recommended to use the project virtual environment:

```powershell
.\myenv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data Setup

By default, the pipeline expects EEG data in a folder named `chb01`:

```text
./chb01
```

The current configuration is based on CHB-MIT style EDF files and seizure annotations for the `chb01` subset. If your EDF filenames or seizure intervals differ, update the mappings in [eeg_pipeline/config.py](eeg_pipeline/config.py).

## How to Run

From the project root, run:

```powershell
python eeg_pipeline/main.py --data_dir ./chb01
```

You can skip specific tasks if needed:

```powershell
python eeg_pipeline/main.py --data_dir ./chb01 --skip_tasks cnn rf
```

Valid task flags are:

- `pred` for seizure prediction
- `cnn` for disease classification
- `rf` for random forest diagnosis
- `ae` for anomaly detection

## Output Artifacts

The pipeline writes generated artifacts to these folders:

- `results/` for plots and evaluation outputs
- `saved_models/` for trained model checkpoints
- `logs/` for runtime logs

These folders are created automatically by [eeg_pipeline/config.py](eeg_pipeline/config.py).

## Notes

- The codebase is structured as a research pipeline and is intended to be extended with real training logic, data splits, and dataset-specific annotation handling.
- The `main.py` entry point orchestrates the full workflow end to end.
- If you change the dataset layout or channel names, update [eeg_pipeline/config.py](eeg_pipeline/config.py) first.

## Suggested Next Steps

1. Add `__init__.py` files so `eeg_pipeline` can be imported as a package.
2. Add starter implementations to each module if you want the scaffold to run immediately.
3. Add a sample data loader or example notebook for CHB-MIT EEG records.
