# S_FE Payload Feature Pipeline

This fork extends the original flow-statistics pipeline with the **five-feature
S_FE payload vector** described in the paper *Efficient Feature
Engineering-Based Anomaly Detection for Network Security*.

## 1. Build the dataset

```bash
# Install deps (pyshark, mmh3, scikit-learn, numpy, etc.)
pip install -r requirements.txt

# Parse PCAPs inside ./data/ and save compressed NumPy arrays
python -m src.build_sfe_dataset \
    --pcap_dir data \
    --out data/sfe_offline.npz
```

*Tip:* Start with a single PCAP (e.g. Friday) to validate the pipeline before
processing all days (≈ 60 GB).

## 2. Train / evaluate

```bash
# Payload-based experiment
python -m src.train_flow_gpu \
    --payload_npz data/sfe_offline.npz \
    --output results_payload

# Flow-stat baseline (old behaviour)
python -m src.train_flow_gpu \
    --csv data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
    --output results_flow
```

After running, inspect metrics with

```bash
python -m src.analyze_results --dir results_payload
```

## Feature order

```
trust_value, byte_freq_entropy, byte_entropy, payload_len, stream_idx
```

## Notes
* Trust value is computed per-source-IP **before** updating counters to match
  the paper.
* Attack intervals for CIC-IDS 2017 are hard-coded in
  `[src/build_sfe_dataset.py](src/build_sfe_dataset.py)`. Edit if you have a
  refined schedule.
* The script requires *pyshark* which calls out to `tshark` – install Wireshark
  or `sudo apt install tshark` if missing.

