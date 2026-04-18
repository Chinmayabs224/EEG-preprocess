# SNN-Encoded Ensemble Model for EEG Seizure Detection

> An end-to-end, production-grade machine learning pipeline for processing raw EEG brainwaves, converting them into compact Spiking Neural Network (SNN) features, and utilizing an ensemble of four diverse Deep Learning architectures for robust seizure detection.

---

## 📌 Project Architecture

This project was built to efficiently handle the massive **CHB-MIT Scalp EEG Database**, mitigating extreme class imbalances (~99.5% normal vs ~0.5% seizure data) through hybrid machine learning strategies.

The pipeline comprises two main components separated for memory efficiency and scalability:

### 1. The Preprocessing Pipeline (`spike_pipeline/`)
Because raw EEG data is massive and extremely noisy, we compress the data before training any neural networks.
* **Filtering:** Applies 50/60Hz Notch filters, 0.5–40Hz bandpass filtering, and Common Average Referencing (CAR).
* **Chunking:** Parses the continuous `.edf` sequences into distinct 60-second windows.
* **SNN Encoding:** Instead of feeding raw amplitudes into a model, we convert the voltages into biological "spikes" using a **Spiking Neural Network (SNN)** approach. This compresses the data into 64 highly-informative spike-rate features per window.
* **Storage:** Outputs compressed `.npz` files (one per recording) containing `snn_features` and binary `snn_labels`.

### 2. The Downstream Ensemble Pipeline (`training/`)
Once the `.npz` features are generated, four distinct machine learning architectures are trained. They each possess a unique mathematical perspective on how to identify a seizure:

1. **Autoencoder (Unsupervised Anomaly Detection):**
   * *How it works:* Trained **exclusively** on healthy/normal brainwaves. It acts as an anomaly detector. When a seizure is fed into the Autoencoder, it fails to reconstruct the signal properly, causing a huge "Reconstruction Error" spike, signaling an event.
2. **1D CNN (Window-Level Classifier):**
   * *How it works:* Treats each window completely independently. Excellent at finding localized spatial/frequency patterns across the 64 SNN channels without needing temporal context.
3. **BiLSTM + Attention (Sequential Temporal Classifier):**
   * *How it works:* Processes the data in 30-window sequences (representing 30 minutes of context). It learns how seizures physically evolve over time, using an Attention mechanism to focus on the moments right before the seizure strikes.
4. **Transformer Encoder (Multi-head Attention):**
   * *How it works:* Uses multi-head self-attention and sinusoidal positional embeddings to discover complex, long-range correlations between brain states across extended periods of time.
5. **Weighted Ensemble Layer:**
   * *How it works:* An intelligent post-processing layer that takes the probability outputs from all 4 models and runs a Grid Search threshold optimization. It calculates the mathematically perfect blend of votes to maximize the F1 Score and eliminate False Positives.

---

## 🚀 How to Run the Pipeline (AWS EC2 / Ubuntu)

### Step 1: Preprocess the Raw EDF files
Extract features from raw CHB-MIT data and convert them into compressed `.npz` arrays.
```bash
python3 -m spike_pipeline.run_pipeline \
    --data_dir /path/to/raw/edf/files \
    --output_dir ./output/spikes
```

### Step 2: Train the ML Models
Because of the heavy CPU loads on 1.2+ million windows, use `nohup` to safely run the background orchestration script, which handles automatic dataset stratification and trains all 4 models sequentially.
```bash
nohup python3 -m training.train_all \
    --spikes_dir ./output/spikes \
    --save_dir ./models \
    --plots_dir ./training_plots \
    --epochs 50 \
    --batch_size 64 \
    --seq_len 30 > training.log 2>&1 &
```
*Monitor live progress:* `tail -f training.log`

### Step 3: Run Inference & Visualizations
Test the trained models on new unseen data to generate ROC Curves, Timeline Probability tracking, and automated model-comparison graphs.
```bash
python3 -m training.predict \
    --spikes_dir ./output/spikes \
    --models_dir ./models \
    --plots_dir ./test_plots
```

---

## 🧬 Handling Clinical Class Imbalance

EEG data is notoriously imbalanced. A patient might be monitored for 24 hours but only have a 30-second seizure. To combat this mathematical distortion during training, the pipeline explicitly uses:
1. **WeightedRandomSampler:** During training, PyTorch forces the dataloader to sample seizures heavily so the model sees a balanced 50/50 mix. (Without this, the models achieve 99% accuracy by simply guessing "Normal" constantly).
2. **Class-Weighted Cross Entropy:** Punishes the models much harder for missing a real seizure (False Negative) than for throwing a false alarm.
3. **F1 / AUC-ROC Metrics:** The pipeline abandons standard Accuracy, instead grading models based on the Area Under the Receiver Operating Characteristic curve.

---

## 📂 Project Directory Map

```text
EEG-preprocess/
├── spike_pipeline/
│   ├── config.py             # SNN parameters, thresholds, channel exclusion
│   ├── data_loader.py        # CPU-safe generator for raw EDF parsing
│   ├── preprocessing.py      # Artifact removal, Notch, CAR
│   ├── snn_feature_extractor.py # Spiking Neural Net encoder
│   └── run_pipeline.py       # Orchestrator
│
├── training/
│   ├── dataset.py            # Lazy-slicing Dataloader to prevent RAM OOM
│   ├── model_autoencoder.py  # FC Reconstruction Model
│   ├── model_cnn.py          # 1D Convolutional Neural Network
│   ├── model_lstm.py         # Bidirectional LSTM + Additive Attention
│   ├── model_transformer.py  # Transformer Encoder
│   ├── ensemble.py           # Weighted Probability Voting Engine
│   ├── utils.py              # EarlyStopping & ROC Visualizers
│   ├── train_all.py          # Training Orchestrator CLI
│   └── predict.py            # Inference & Testing CLI
│
└── requirements.txt          # Python dependencies (MNE, PyTorch, Scikit etc.)
```
