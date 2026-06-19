"""
Synthetic CICIDS2017-like dataset generator.
Generates realistic network traffic data mimicking the CICIDS2017 dataset
for training the intrusion detection system when the real dataset is unavailable.
"""
import numpy as np
import pandas as pd
import os
from src.config import SYNTHETIC_DATASET_PATH, DATASET_PATH


def generate_synthetic_cicids2017(n_samples=10000, seed=42):
    """
    Generate a synthetic dataset resembling CICIDS2017 structure.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with CICIDS2017-like features and labels
    """
    rng = np.random.default_rng(seed)

    data = pd.DataFrame()

    # Flow duration (microseconds): normal traffic tends to have varied durations,
    # attacks often have shorter or specific patterns
    data["Flow Duration"] = rng.exponential(scale=50000, size=n_samples).astype(int)
    data["Flow Duration"] = data["Flow Duration"].clip(1, 10000000)

    # Total Fwd Packets
    data["Total Fwd Packets"] = rng.negative_binomial(5, 0.3, size=n_samples) + 1
    data["Total Fwd Packets"] = data["Total Fwd Packets"].clip(1, 50000)

    # Total Backward Packets
    data["Total Backward Packets"] = rng.negative_binomial(3, 0.4, size=n_samples)
    data["Total Backward Packets"] = data["Total Backward Packets"].clip(0, 30000)

    # Total Length of Fwd Packets
    data["Total Length of Fwd Packets"] = (
        data["Total Fwd Packets"] * rng.exponential(scale=500, size=n_samples)
    ).astype(int)
    data["Total Length of Fwd Packets"] = data["Total Length of Fwd Packets"].clip(0, 10000000)

    # Total Length of Bwd Packets
    data["Total Length of Bwd Packets"] = (
        data["Total Backward Packets"] * rng.exponential(scale=300, size=n_samples)
    ).astype(int)
    data["Total Length of Bwd Packets"] = data["Total Length of Bwd Packets"].clip(0, 5000000)

    # Fwd Packet Length metrics
    data["Fwd Packet Length Mean"] = rng.exponential(scale=400, size=n_samples) + 20
    data["Fwd Packet Length Max"] = data["Fwd Packet Length Mean"] * rng.uniform(1.5, 5, size=n_samples)
    data["Fwd Packet Length Min"] = data["Fwd Packet Length Mean"] * rng.uniform(0.1, 0.8, size=n_samples)
    data["Fwd Packet Length Std"] = rng.exponential(scale=200, size=n_samples) + 10

    # Bwd Packet Length metrics
    data["Bwd Packet Length Mean"] = rng.exponential(scale=250, size=n_samples) + 10
    data["Bwd Packet Length Max"] = data["Bwd Packet Length Mean"] * rng.uniform(1.5, 5, size=n_samples)
    data["Bwd Packet Length Min"] = data["Bwd Packet Length Mean"] * rng.uniform(0.1, 0.8, size=n_samples)

    # Flow related features
    data["Flow Bytes/s"] = np.where(
        data["Flow Duration"] > 0,
        (data["Total Length of Fwd Packets"] + data["Total Length of Bwd Packets"])
        / (data["Flow Duration"] / 1_000_000),
        0
    )
    data["Flow Packets/s"] = np.where(
        data["Flow Duration"] > 0,
        (data["Total Fwd Packets"] + data["Total Backward Packets"])
        / (data["Flow Duration"] / 1_000_000),
        0
    )

    # Protocol (6=TCP, 17=UDP, 1=ICMP)
    protocol_choices = rng.choice([6, 17, 1], size=n_samples, p=[0.7, 0.2, 0.1])
    data["Protocol"] = protocol_choices

    # Ports
    data["Source Port"] = rng.integers(1024, 65535, size=n_samples)
    # Destination port: common services
    common_ports = [80, 443, 22, 21, 25, 53, 110, 143, 3389, 8080, 445, 135]
    data["Destination Port"] = rng.choice(common_ports, size=n_samples)
    # Mix in some random high ports
    high_port_mask = rng.random(size=n_samples) < 0.3
    data.loc[high_port_mask, "Destination Port"] = rng.integers(1024, 65535, size=high_port_mask.sum())

    # Additional CICIDS2017 features for realism
    data["Fwd IAT Mean"] = rng.exponential(scale=1000, size=n_samples)
    data["Fwd IAT Std"] = rng.exponential(scale=500, size=n_samples)
    data["Bwd IAT Mean"] = rng.exponential(scale=1500, size=n_samples)
    data["Bwd IAT Std"] = rng.exponential(scale=700, size=n_samples)
    data["Fwd PSH Flags"] = rng.binomial(1, 0.1, size=n_samples)
    data["Fwd URG Flags"] = rng.binomial(1, 0.02, size=n_samples)
    data["FIN Flag Count"] = rng.poisson(0.5, size=n_samples)
    data["SYN Flag Count"] = rng.poisson(2, size=n_samples)
    data["RST Flag Count"] = rng.poisson(0.3, size=n_samples)
    data["PSH Flag Count"] = rng.poisson(0.5, size=n_samples)
    data["ACK Flag Count"] = rng.poisson(5, size=n_samples)
    data["URG Flag Count"] = rng.poisson(0.1, size=n_samples)
    data["CWE Flag Count"] = rng.poisson(0.05, size=n_samples)
    data["ECE Flag Count"] = rng.poisson(0.05, size=n_samples)
    data["Average Packet Size"] = rng.exponential(scale=400, size=n_samples) + 40
    data["Avg Fwd Segment Size"] = data["Fwd Packet Length Mean"]
    data["Avg Bwd Segment Size"] = data["Bwd Packet Length Mean"]
    data["Subflow Fwd Packets"] = data["Total Fwd Packets"] * rng.uniform(0.8, 1.0, size=n_samples)
    data["Subflow Fwd Bytes"] = data["Total Length of Fwd Packets"] * rng.uniform(0.8, 1.0, size=n_samples)
    data["Subflow Bwd Packets"] = data["Total Backward Packets"] * rng.uniform(0.8, 1.0, size=n_samples)
    data["Subflow Bwd Bytes"] = data["Total Length of Bwd Packets"] * rng.uniform(0.8, 1.0, size=n_samples)
    data["Init Fwd Win Bytes"] = rng.integers(1000, 65535, size=n_samples)
    data["Init Bwd Win Bytes"] = rng.integers(1000, 65535, size=n_samples)
    data["Fwd Act Data Pkts"] = data["Total Fwd Packets"] * rng.uniform(0.7, 1.0, size=n_samples)
    data["Fwd Seg Size Min"] = rng.integers(20, 200, size=n_samples)
    data["Active Mean"] = rng.exponential(scale=5000, size=n_samples)
    data["Active Std"] = rng.exponential(scale=2000, size=n_samples)
    data["Idle Mean"] = rng.exponential(scale=10000, size=n_samples)
    data["Idle Std"] = rng.exponential(scale=3000, size=n_samples)

    # Round float columns for cleaner data
    float_cols = data.select_dtypes(include=[np.float64]).columns
    data[float_cols] = data[float_cols].round(6)
    int_cols = data.select_dtypes(include=[np.int32, np.int64]).columns
    data[int_cols] = data[int_cols].astype(int)

    # ---- Generate labels ----
    # Create attack patterns based on feature combinations
    labels = ["BENIGN"] * n_samples
    labels = np.array(labels)

    # Attack pattern 1: DoS - high packet count, low duration, many SYN flags
    dos_condition = (
        (data["Total Fwd Packets"] > 100) &
        (data["Flow Duration"] < 100000) &
        (data["SYN Flag Count"] > 3) &
        (data["Protocol"] == 6)  # TCP
    )
    labels[dos_condition] = "DoS"

    # Attack pattern 2: Port Scan - many connections to different ports, low duration
    scan_condition = (
        (data["Total Fwd Packets"] < 20) &
        (data["Flow Duration"] < 50000) &
        (data["Destination Port"].isin([22, 23, 3389, 445, 135])) &
        (data["Protocol"] == 6)
    )
    labels[scan_condition] = "PortScan"

    # Attack pattern 3: DDoS - very high packets, high bytes, short duration
    ddos_condition = (
        (data["Total Fwd Packets"] > 500) &
        (data["Total Length of Fwd Packets"] > 500000) &
        (data["Flow Duration"] < 200000) &
        (data["Protocol"] == 6)
    )
    labels[ddos_condition] = "DDoS"

    # Attack pattern 4: Brute Force - SSH/FTP on common ports
    brute_force_condition = (
        (data["Total Fwd Packets"].between(30, 200)) &
        (data["Destination Port"].isin([22, 21])) &
        (data["Protocol"] == 6) &
        (data["SYN Flag Count"] > 2)
    )
    labels[brute_force_condition] = "BruteForce"

    # Attack pattern 5: Botnet traffic - specific packet patterns
    botnet_condition = (
        (data["Total Fwd Packets"].between(5, 50)) &
        (data["Bwd Packet Length Mean"] < 100) &
        (data["Protocol"] == 6) &
        (data["Destination Port"] == 443)
    )
    labels[botnet_condition] = "Botnet"

    # Attack pattern 6: ICMP flood
    icmp_condition = (
        (data["Protocol"] == 1) &
        (data["Total Fwd Packets"] > 50) &
        (data["Flow Duration"] < 50000)
    )
    labels[icmp_condition] = "ICMP_Flood"

    # Attack pattern 7: Data exfiltration - large outgoing traffic
    exfil_condition = (
        (data["Total Length of Fwd Packets"] > 1000000) &
        (data["Flow Duration"] > 500000) &
        (data["Destination Port"].isin([80, 443]))
    )
    labels[exfil_condition] = "Exfiltration"

    # Attack pattern 8: Heartbleed/SSL attack
    ssl_attack_condition = (
        (data["Total Fwd Packets"].between(1, 10)) &
        (data["Total Length of Fwd Packets"].between(100, 5000)) &
        (data["Destination Port"] == 443) &
        (data["Protocol"] == 6)
    )
    labels[ssl_attack_condition] = "SSL_Attack"

    # Convert to binary labels: BENIGN -> 0, any attack -> 1
    binary_labels = np.where(labels == "BENIGN", 0, 1)

    data["Label"] = labels
    data["Binary Label"] = binary_labels

    # Shuffle the dataset
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Ensure some minimum attack ratio (aim for ~25-35% attacks)
    current_attack_ratio = data["Binary Label"].mean()
    target_attack_ratio = 0.28

    if current_attack_ratio < target_attack_ratio:
        # Need more attacks - flip some benign samples to attack
        benign_mask = data["Binary Label"] == 0
        benign_indices = data[benign_mask].index
        n_needed = int(n_samples * target_attack_ratio) - int(data["Binary Label"].sum())
        if n_needed > 0 and len(benign_indices) > 0:
            flip_indices = rng.choice(benign_indices, size=min(n_needed, len(benign_indices)), replace=False)
            data.loc[flip_indices, "Label"] = rng.choice(
                ["DoS", "PortScan", "DDoS", "BruteForce", "Botnet", "ICMP_Flood"],
                size=len(flip_indices)
            )
            data.loc[flip_indices, "Binary Label"] = 1

    return data


def main():
    """Generate and save the synthetic dataset."""
    print("[INFO] Generating synthetic CICIDS2017-like dataset...")
    df = generate_synthetic_cicids2017(n_samples=10000)
    df.to_csv(SYNTHETIC_DATASET_PATH, index=False)

    # Also save to the generic dataset path for easier access
    df.to_csv(DATASET_PATH, index=False)

    n_attacks = df["Binary Label"].sum()
    n_normal = len(df) - n_attacks
    print(f"[INFO] Generated {len(df)} samples:")
    print(f"       Normal: {n_normal} ({n_normal/len(df)*100:.1f}%)")
    print(f"       Attacks: {n_attacks} ({n_attacks/len(df)*100:.1f}%)")
    print(f"[INFO] Saved to: {SYNTHETIC_DATASET_PATH}")
    print(f"[INFO] Saved to: {DATASET_PATH}")
    print(f"[INFO] Columns: {list(df.columns[:20])}...")


if __name__ == "__main__":
    main()
