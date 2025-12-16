# src/build_sfe_dataset.py
"""Build the offline S_FE 5-feature dataset from CIC-IDS 2017 PCAPs.

Example usage
-------------
python -m src.build_sfe_dataset \
    --pcap_dir data/ \
    --out data/sfe_offline.npz

The script expects the PCAP files (Monday-…p and so on) to reside in
*pcap_dir*. A minimal timestamp-to-label mapping for the 2017 dataset is
embedded, but you can supply your own CSV via ``--schedule`` if you have a
more precise table.

Important: running on all days (~60 GB) can take 30-40 minutes on a modern
laptop. Start with one or two PCAPs (e.g., Friday) to verify the pipeline.
"""
from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np
import pandas as pd

# -------------------- Scapy --------------------
try:
    from scapy.all import (
        RawPcapReader,
        PcapReader,
        Ether,
        IP,
        IPv6,
        TCP,
        UDP,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Scapy is required. Run `pip install scapy`. ") from exc

ETH_LEN = 14  # Ethernet header length

# Local import (supports both `python src/...` and `python -m src...`)
try:
    from payload_features import make_feature_row  # when cwd==project root
except ImportError:
    from src.payload_features import make_feature_row

# ---------------------------------------------------------------------------
# CIC-IDS 2017 attack schedule (simplified) – start/end in UNIX epoch seconds
# ---------------------------------------------------------------------------
# These intervals were transcribed from the dataset README and papers; adjust
# if you have a more precise list.
_CIC_SCHEDULE: Dict[str, List[Tuple[int, int]]] = {
    "Monday": [],  # no attacks
    "Tuesday": [
        (1491903600, 1491906000),  # FTP-Patator
        (1491906000, 1491908400),  # SSH-Patator
    ],
    "Thursday": [
        (1492072800, 1492076400),  # DoS Hulk
        (1492076400, 1492080000),  # DoS GoldenEye
        (1492080000, 1492083600),  # Heartbleed + slowloris
    ],
    "Friday": [
        (1492155600, 1492159200),  # DDoS LOIT
    ],
}


# ---------------------------------------------------------------------------
# Helper – label by timestamp
# ---------------------------------------------------------------------------

def label_for_timestamp(ts: float, day_key: str) -> int:
    """Return 0 (benign) or 1 (attack) for *ts* given schedule of *day_key*."""
    for start, end in _CIC_SCHEDULE.get(day_key, []):
        if start <= ts <= end:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------

def process_pcap(pcap_path_str: str, csv_dir: Path) -> Tuple[List[List[float]], List[int]]:
    """Process one PCAP with Scapy and return (features, labels)."""
    pcap_path = Path(pcap_path_str)
    print(f"[+] Loading {pcap_path.name} …")
    day_key = _day_key_from_name(pcap_path.name)

    # CSV-based label map: collect attack time intervals (start,end)
    attack_intervals: List[Tuple[float, float]] = []
    csv_candidates = list(csv_dir.glob(f"*{day_key}*.csv"))
    print(f"[DEBUG] day_key={day_key}, csv_candidates={csv_candidates}")
    if csv_candidates:
        # prefer file without "(2)" suffix (the correct GeneratedLabelFlows version)
        csv_path = next((c for c in csv_candidates if "(2)" not in c.name), csv_candidates[0])
        print(f"[DEBUG] using CSV: {csv_path.name}")
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()

        needed = {"Timestamp", "Flow Duration", "Label"}
        if needed <= set(df.columns):
            df = df[list(needed)]
            df["Label"] = df["Label"].str.strip()
            df = df[df["Label"].str.upper() != "BENIGN"]
        else:
            df = pd.DataFrame()

        if not df.empty:
            # initial interval list (CSV local time)
            starts = pd.to_datetime(df["Timestamp"]).astype("int64") / 1e9
            durations = df["Flow Duration"].astype(float) / 1e6
            intervals_csv = list(zip(starts, starts + durations))

            # compute clock offset: first packet ts minus first CSV ts
            first_pkt_ts = float(next(PcapReader(str(pcap_path))).time)
            offset = first_pkt_ts - intervals_csv[0][0]

            attack_intervals = sorted([(s+offset, e+offset) for s,e in intervals_csv])
            print(f"[i] {csv_path.name}: {len(attack_intervals)} intervals, offset {offset/3600:.2f} h")

    # Trust tracker (rate-based heuristic to avoid data leakage)
    # Map: IP -> (start_ts, packet_count)
    ip_trust_stats: Dict[str, Tuple[float, int]] = {}

    current_flow_idx: Dict[str, int] = defaultdict(int)

    X_rows: List[List[float]] = []
    y_rows: List[int] = []

    pkt_iter = PcapReader(str(pcap_path))  # robust parser
    total_seen = skipped_not_ip = skipped_no_ts = 0
    for scapy_pkt in pkt_iter:
        pkt_data = bytes(scapy_pkt)
        ts = float(scapy_pkt.time)
        total_seen += 1

        # minimal Ethernet + IPv4 parsing
        if len(pkt_data) < ETH_LEN + 20:  # too short
            continue
        ip_layer = scapy_pkt.getlayer(IP) or scapy_pkt.getlayer(IPv6)
        if ip_layer is None:
            skipped_not_ip += 1
            if total_seen % 100_000 == 0:
                print(
                    f"  {pcap_path.name}: {total_seen/1e6:.1f}M frames – "
                    f"{len(X_rows):,} samples, {skipped_not_ip:,} non-IP, {skipped_no_ts:,} no-ts"
                )
            continue

        src = ip_layer.src
        dst = ip_layer.dst
        # protocol number differs between IPv4 (proto) and IPv6 (nh)
        if hasattr(ip_layer, "proto"):
            proto = ip_layer.proto  # IPv4
        else:
            proto = ip_layer.nh  # IPv6

        if scapy_pkt.haslayer(TCP):  # TCP
            sport = scapy_pkt[TCP].sport
            dport = scapy_pkt[TCP].dport
            payload_bytes = bytes(scapy_pkt[TCP].payload)
        elif scapy_pkt.haslayer(UDP):  # UDP
            sport = scapy_pkt[UDP].sport
            dport = scapy_pkt[UDP].dport
            payload_bytes = bytes(scapy_pkt[UDP].payload)
        else:
            sport = dport = 0
            payload_bytes = bytes(ip_layer.payload)

        key = f"{src}:{sport}>{dst}:{dport}/{proto}"

        # 1. Trust Value (Heuristic: Traffic Rate)
        # High packet rate -> Low Trust.
        # This replaces the previous "Ground Truth" leakage.
        if src not in ip_trust_stats:
            ip_trust_stats[src] = (ts, 0)
        
        t_start, t_count = ip_trust_stats[src]
        duration = ts - t_start
        rate = t_count / max(duration, 0.1) # avoid div by zero
        # Decay trust as rate increases. e.g. 100 pps -> ~0.5
        trust_val = 1.0 / (1.0 + (rate / 100.0))
        
        ip_trust_stats[src] = (t_start, t_count + 1)

        # 2. Direction (0=Outgoing/Internal, 1=Incoming/External)
        # CIC-IDS 2017 internal subnet is 192.168.10.x
        is_internal = src.startswith("192.168.")
        direction = 0 if is_internal else 1

        stream_idx = current_flow_idx[key]
        current_flow_idx[key] += 1

        # interval-based label
        if attack_intervals:
            idx = bisect.bisect_left(attack_intervals, (ts, ts))
            is_attack = False
            for j in (idx - 1, idx):
                if 0 <= j < len(attack_intervals):
                    s_, e_ = attack_intervals[j]
                    if s_ <= ts <= e_:
                        is_attack = True
                        break
            label = 1 if is_attack else 0
        else:
            label = 0

        X_rows.append(
            make_feature_row(
                payload_bytes,
                trust_value=trust_val,
                stream_idx=stream_idx,
                direction=direction,
                use_hash=True,
            )
        )
        y_rows.append(label)

        # Update stats (already done above for Trust)
        pass

        if total_seen % 1_000_000 == 0:
            print(
                f"  {pcap_path.name}: {total_seen/1e6:.0f}M frames – "
                f"{len(X_rows):,} samples, {skipped_not_ip:,} non-IP, {skipped_no_ts:,} no-ts"
            )

    pkt_iter.close()

    return X_rows, y_rows


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _day_key_from_name(fname: str) -> str:
    for key in _CIC_SCHEDULE.keys():
        if fname.lower().startswith(key.lower()):
            return key
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Build S_FE dataset from PCAPs")
    ap.add_argument("--pcap_dir", default="data", help="directory with *.pcap files")
    ap.add_argument("--out", default="data/sfe_offline.npz", help="output .npz path")
    ap.add_argument("--workers", type=int, default=1, help="parallel workers (>1 for multi-core)")
    ap.add_argument("--files", help="comma-separated list of PCAP filenames to process (optional)")
    args = ap.parse_args()

    pcap_dir = Path(args.pcap_dir)
    out_path = Path(args.out)

    X_all: List[List[float]] = []
    y_all: List[int] = []

    if args.files:
        allow = {f.strip() for f in args.files.split(",")}
        pcap_files: Sequence[Path] = [pcap_dir / f for f in allow]
    else:
        pcap_files = sorted(pcap_dir.glob("*.pcap"))

    if args.workers == 1:
        for p in pcap_files:
            X_rows, y_rows = process_pcap(str(p), pcap_dir)
            X_all.extend(X_rows)
            y_all.extend(y_rows)
            print(f"  → {p.name}: {len(y_rows):,} samples")

            # incremental save after each PCAP
            X = np.asarray(X_all, dtype=np.float32)
            y = np.asarray(y_all, dtype=np.int8)
            np.savez_compressed(out_path, X=X, y=y, feature_names=np.array([
                "trust_value",
                "byte_freq_std",
                "byte_entropy",
                "payload_len",
                "stream_idx",
                "direction",
                "hash_value",
            ]))
            print(f"    [saved interim dataset – {X.shape[0]:,} samples]")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"[+] Processing {len(pcap_files)} PCAPs with {args.workers} workers …")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_pcap, str(p), pcap_dir): p for p in pcap_files}
            for fut in as_completed(futs):
                p = futs[fut]
                X_rows, y_rows = fut.result()
                X_all.extend(X_rows)
                y_all.extend(y_rows)
                print(f"  → {p.name}: {len(y_rows):,} samples")

    X = np.asarray(X_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.int8)

    print(f"[+] Saving dataset to {out_path} – shape {X.shape}")
    np.savez_compressed(out_path, X=X, y=y, feature_names=np.array([
        "trust_value",
        "byte_freq_std",
        "byte_entropy",
        "payload_len",
        "stream_idx",
        "direction",
        "hash_value",
    ]))

    meta = {
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "pcap_files": [p.name for p in pcap_dir.glob("*.pcap")],
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    print("[✓] Done.")


if __name__ == "__main__":
    main()

