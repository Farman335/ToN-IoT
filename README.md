# Sentinel-IoT: An Autonomous and Explainable Extended Detection and Response Framework for Heterogeneous Industrial IoT Security and Compliance Auditing

Official implementation of the **Sentinel-IoT** framework for autonomous edge intrusion detection, real-time forensic attack classification, sub-millisecond first-order gradient explainability, automated Wazuh active mitigation, and continuous GRC compliance auditing across heterogeneous Industrial IoT (IIoT) ecosystems.

---

## 📋 Architectural Overview

Sentinel-IoT addresses cross-layer cyber-physical attack surfaces in smart manufacturing and critical infrastructures by combining:
- **Conformer-Sentinel Neural Core**: Interleaves Depthwise Separable Convolutions with Multi-Head Self-Attention (MHSA) in a Macaron sandwich structure to concurrently capture high-frequency physical sensor bursts and long-range sequential attack patterns across a sliding temporal buffer ($T=10$).
- **Parallel Dual-Head Topology**: Jointly optimizes an auxiliary binary anomaly detection head ($\mathcal{L}_{\text{BCE}}$) and a forensic multi-class classification head ($\mathcal{L}_{\text{CE}}$).
- **Sub-Millisecond First-Order Saliency Attribution**: Direct mathematical gradient attribution derived from the computational graph in $0.30\,\text{ms}$ ($90.85\%$ faster than perturbation-based Kernel SHAP).
- **Closed-Loop Active Mitigation**: Automated containment (Wazuh active response scripts including `iptables` ingress drops and rogue process termination) executed within an empirical Mean Time to Respond ($\text{MTTR} = 22.69\,\text{ms}$).
- **Automated GRC Control Mapping**: Maps telemetry anomalies to NIST SP 800-53 Rev. 5 and ISO/IEC 27001:2022 regulatory controls.

---

## 🗂️ Clean Modular Code Organization

This package contains the essential, curated, and modularized Python implementation files:

| Renamed Script | Description |
| :--- | :--- |
| **`01_data_preprocessing.py`** | Multi-domain dataset cleaning pipeline: handles missing values, encodes categorical state variables, isolates audit identifiers, performs min-max feature scaling, and constructs sliding sequence tensors ($T=10$). |
| **`02_conformer_model_and_training.py`** | PyTorch implementation of the complete Conformer-Sentinel neural network (Macaron FeedForward, Depthwise Separable Conv, MHSA), dual-head joint loss objective, and multi-epoch training loop. |
| **`03_edge_xdr_inference_gateway.py`** | Real-time FastAPI edge inference gateway with circular sliding buffers, sub-millisecond analytical saliency feature attribution, closed-loop Wazuh mitigation triggering, and automated NIST/ISO GRC logging. |
| **`04_mqtt_telemetry_streamer.py`** | Lightweight MQTT telemetry ingestion client simulating edge sensor telemetry streaming across industrial broker endpoints. |
| **`05_baseline_models_benchmark.py`** | Benchmark evaluation pipeline comparing Conformer-Sentinel against deep learning baselines (BiLSTM, GRU, 1D-CNN) across identical multi-domain partitions. |
| **`06_architectural_ablation_study.py`** | Empirical ablation framework evaluating the necessity of depthwise convolutions, self-attention, temporal sequence buffering ($T=10$ vs. $T=1$), and dual-head joint anomaly supervision. |
| **`07_statistical_significance_testing.py`** | Formal hypothesis testing adhering to Demšar's protocol: paired Student's $t$-tests, Cohen's $d$, Wilcoxon signed-rank tests, Friedman omnibus ranking, and Nemenyi Critical Difference ($\text{CD} = 1.301$). |

---

## 📊 Dataset Access & Links

The framework was evaluated across all **13 heterogeneous sub-datasets** of the benchmark suite spanning:
1. **Network Flows**: Network Packet Traffic (Zeek/Bro connection logs)
2. **Physical IoT Sensors (7 domains)**: IoT Fridge, IoT GPS Tracker, IoT Garage Door, IoT Modbus, IoT Motion Light, IoT Thermostat, IoT Weather Station
3. **Host Operating Systems (5 domains)**: Linux Disk, Linux Memory, Linux Process, Windows 10, Windows 7

### Official Dataset Repositories:
- **GitHub Repository**: [https://github.com/Farman335/ToN-IoT](https://github.com/Farman335/ToN-IoT)
- **IEEE DataPort / UNSW Canberra**: [ToN_IoT Datasets](https://ieee-dataport.org/documents/toniot-datasets)
- **UNSW Canberra Research**: [Cyber Range and IoT Testbed Datasets](https://research.unsw.edu.au/projects/toniot-datasets)

---

## 🚀 Quickstart & Installation

### 1. Environment Setup
```bash
git clone https://github.com/Farman335/ToN-IoT.git
cd ToN-IoT
pip install -r requirements.txt
```

### 2. Preprocess Raw Telemetry
```bash
python 01_data_preprocessing.py
```

### 3. Train the Conformer-Sentinel Model
```bash
python 02_conformer_model_and_training.py
```

### 4. Launch Edge XDR Inference Gateway
```bash
python 03_edge_xdr_inference_gateway.py
```
The FastAPI edge endpoint will be available at `http://localhost:8000/docs`.

### 5. Simulate Real-Time MQTT Telemetry
```bash
python 04_mqtt_telemetry_streamer.py
```

### 6. Run Baselines, Ablations, & Statistical Tests
```bash
# Evaluate baseline comparisons (BiLSTM, GRU, 1D-CNN)
python 05_baseline_models_benchmark.py

# Run component ablation experiments
python 06_architectural_ablation_study.py

# Execute Demšar statistical significance protocol
python 07_statistical_significance_testing.py
```

---

## ⚡ Empirical Hardware Benchmarks (Edge Gateway)

- **Weight Footprint**: $2.83\,\text{MB}$ ($7.07 \times 10^5$ to $7.22 \times 10^5$ parameters in float32)
- **Peak RAM Allocation**: $4.32\,\text{MB}$
- **Mean CPU Inference Latency**: $4.25\,\text{ms}$ on quad-core ARM Cortex-A72
- **First-Order Saliency Latency**: $0.30\,\text{ms}$ ($90.85\%$ lower than Kernel SHAP at $3.28\,\text{ms}$)
- **Mean Time to Respond (MTTR)**: $22.69\,\text{ms}$ closed-loop automated containment

---

## 📜 Citation & License

This project is licensed under the MIT License. If you utilize Sentinel-IoT or the associated evaluation code, please cite our work:

```bibtex
@unpublished{sentinel_iot_2026,
  title={Sentinel-IoT: An Autonomous and Explainable Extended Detection and Response Framework for Heterogeneous Industrial IoT Security and Compliance Auditing},
  author={Tariq, Wajahat Ali and Hashmi, Abdullah Bin Zubair and Ali, Farman and Hashmi, Muhammad Usman and Alsini, Raed and Yafoz, Ayman},
  note={Manuscript under review},
  year={2026}
}
```
