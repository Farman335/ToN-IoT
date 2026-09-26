import os
import json
import collections
import subprocess
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="Conformer-Sentinel XDR Inference Gateway")

# ---------------------------------------------------------
# CONFORMER SENTINEL NEURAL NETWORK ARCHITECTURE
# ---------------------------------------------------------
class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.w_1 = nn.Linear(d_model, d_model * expansion_factor)
        self.w_2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return residual + 0.5 * x

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2, 
            groups=d_model
        )
        self.pointwise = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=1
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.ln(x)
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return residual + x

class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, kernel_size=3, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, dropout=dropout)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(d_model, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = FeedForwardModule(d_model, dropout=dropout)
        self.post_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ff1(x)
        residual = x
        x_ln = self.attn_ln(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln)
        x = residual + attn_out
        x = self.conv(x)
        x = self.ff2(x)
        x = self.post_ln(x)
        return x

class ConformerSentinelModel(nn.Module):
    def __init__(self, num_features, d_model=128, nhead=4, num_classes=10, num_layers=2):
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.conformers = nn.ModuleList([
            ConformerBlock(d_model=d_model, nhead=nhead, kernel_size=3, dropout=0.1)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, 64)
        self.dropout = nn.Dropout(0.2)
        self.binary_head = nn.Linear(64, 1)
        self.multiclass_head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        for layer in self.conformers:
            x = layer(x)
        x = torch.mean(x, dim=1)
        x = torch.relu(self.fc(x))
        x = self.dropout(x)
        bin_out = self.binary_head(x)
        mul_out = self.multiclass_head(x)
        return bin_out, mul_out

# ---------------------------------------------------------
# CONFIGURATIONS & GLOBAL INITIALIZATIONS
# ---------------------------------------------------------
PROJECT_DIR = "/home/ubuntu/sentinel_xdr"
WEIGHTS_DIR = os.path.join(PROJECT_DIR, "weights")
METADATA_PATH = os.path.join(WEIGHTS_DIR, "model_metadata.json")

# Load model metadata config
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Missing model configuration at: {METADATA_PATH}")

with open(METADATA_PATH, "r") as f:
    model_metadata = json.load(f)

# In-memory sliding sequence buffers (length = 10) grouped by device
sequence_buffers = collections.defaultdict(lambda: collections.deque(maxlen=10))

# Load all PyTorch models into memory
loaded_models = {}
for model_name, meta in model_metadata.items():
    weight_path = os.path.join(WEIGHTS_DIR, f"sentinel_{model_name}.pth")
    # Fallback check for network traffic file name
    if model_name == "Network_Traffic":
        weight_path = os.path.join(WEIGHTS_DIR, "conformer_sentinel_model_opt.pth")
        
    if os.path.exists(weight_path):
        print(f"Loading weights for: {model_name}...")
        model = ConformerSentinelModel(
            num_features=meta["num_features"],
            num_classes=meta["num_classes"]
        )
        # Load weights on CPU
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        model.eval()
        loaded_models[model_name] = model
    else:
        print(f"Warning: Model weights not found for {model_name} at {weight_path}")

# ---------------------------------------------------------
# NIST SP 800-53 & ISO 27001 COMPLIANCE MAPPINGS
# ---------------------------------------------------------
GRC_MAPPING = {
    "ddos": {
        "nist": "SC-5 (Denial of Service Protection) / SC-7 (Boundary Protection)",
        "iso": "A.8.20 (Network Security)"
    },
    "dos": {
        "nist": "SC-5 (Denial of Service Protection) / SC-7 (Boundary Protection)",
        "iso": "A.8.20 (Network Security)"
    },
    "ransomware": {
        "nist": "SI-3 (Malicious Code Protection)",
        "iso": "A.8.7 (Protection Against Malware)"
    },
    "injection": {
        "nist": "SI-10 (Information Input Validation)",
        "iso": "A.8.7 (Protection Against Malware)"
    },
    "password": {
        "nist": "IA-5 (Authenticator Management)",
        "iso": "A.9.4 (Access Control) / A.8.5 (Secure Authentication)"
    },
    "scanning": {
        "nist": "RA-5 (Vulnerability Monitoring and Scanning)",
        "iso": "A.8.8 (Management of Technical Vulnerabilities)"
    },
    "xss": {
        "nist": "SI-10 (Information Input Validation)",
        "iso": "A.8.7 (Protection Against Malware)"
    },
    "backdoor": {
        "nist": "SI-3 (Malicious Code Protection) / SC-7 (Boundary Protection)",
        "iso": "A.8.20 (Network Security) / A.8.7 (Protection Against Malware)"
    },
    "mitm": {
        "nist": "SC-8 (Transmission Integrity)",
        "iso": "A.8.20 (Network Security) / A.8.24 (Use of Cryptography)"
    }
}

class TelemetryLog(BaseModel):
    device_id: str
    features: List[float]
    source_ip: str = "192.168.1.100"
    process_id: int = 1234

# ---------------------------------------------------------
# ACTIVE RESPONSE REMEDIATION SCRIPT (WAZUH)
# ---------------------------------------------------------
def trigger_active_response(threat_type: str, source_ip: str, pid: int):
    """
    Executes automated shell remediation mimicking Wazuh Active Response.
    """
    remediation_log = ""
    if threat_type in ["ddos", "dos", "scanning", "mitm"]:
        # Block IP using iptables
        cmd = f"sudo iptables -A INPUT -s {source_ip} -j DROP"
        remediation_log = f"Wazuh Active Response: Blocked IP {source_ip} using iptables."
        print(f"[REMEDIATION] {remediation_log}")
        # Note: In production VM, this command is executed. For local user safety we mock print it
        # subprocess.run(cmd.split(), capture_output=True)
    elif threat_type in ["ransomware", "injection", "backdoor", "password"]:
        # Kill the malicious process ID
        cmd = f"sudo kill -9 {pid}"
        remediation_log = f"Wazuh Active Response: Terminated compromised process ID (PID) {pid}."
        print(f"[REMEDIATION] {remediation_log}")
        # subprocess.run(cmd.split(), capture_output=True)
    return remediation_log

# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "running", "loaded_models": list(loaded_models.keys())}

@app.post("/predict/{model_name}")
def predict(model_name: str, payload: TelemetryLog):
    if model_name not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded or not found.")
        
    model = loaded_models[model_name]
    meta = model_metadata[model_name]
    
    # Validate input dimensions
    if len(payload.features) != meta["num_features"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Feature size mismatch. Expected {meta['num_features']}, got {len(payload.features)}"
        )
        
    # Append to device's sliding buffer
    buf = sequence_buffers[payload.device_id]
    buf.append(payload.features)
    
    # If buffer is not full (less than 10 events), return queue status
    if len(buf) < 10:
        return {
            "status": "buffering",
            "buffered_steps": len(buf),
            "required_steps": 10,
            "message": "Collecting sequential time steps."
        }
        
    # Convert buffer sequence to tensor: shape (1, 10, num_features)
    seq_data = list(buf)
    
    # Enable gradient tracking for explainable AI (SHAP-Saliency mapping)
    tensor_input = torch.tensor([seq_data], dtype=torch.float32, requires_grad=True)
    
    bin_logits, mul_logits = model(tensor_input)
    
    # Binary Head (Zero-Day Anomaly Detection)
    anomaly_prob_tensor = torch.sigmoid(bin_logits)
    anomaly_prob = anomaly_prob_tensor.item()
    is_anomaly = anomaly_prob > 0.85
    
    # Compute Saliency/SHAP explainability weights via gradients
    model.zero_grad()
    anomaly_prob_tensor.backward()
    
    import numpy as np
    grads = tensor_input.grad.detach().cpu().numpy()[0] # shape (10, num_features)
    # Average absolute gradient over the 10 time steps to get general feature importance
    mean_abs_grads = np.mean(np.abs(grads), axis=0)
    total_grad = np.sum(mean_abs_grads) + 1e-9
    shap_explainability = (mean_abs_grads / total_grad).tolist()
    
    # Multiclass Head (Forensic Attack Classification)
    with torch.no_grad():
        pred_class_idx = torch.argmax(mul_logits, dim=1).item()
        predicted_class_name = meta["class_names"][pred_class_idx]
        
    # Active response and threshold logic (Section 5.5 of Proposal)
    action_taken = ""
    remediation_status = "none"
    if anomaly_prob > 0.85:
        # Critical Alert - Trigger Wazuh Automated Remediation
        action_taken = trigger_active_response(predicted_class_name, payload.source_ip, payload.process_id)
        remediation_status = "active_blocked"
    elif anomaly_prob >= 0.60:
        # Marginal Warning - Log priority alert but do not block
        action_taken = "High-priority warning alert raised. Human verification required."
        remediation_status = "warning_raised"
        
    # Map GRC Compliance Controls (Section 4.0 of WBS)
    nist_control = "N/A"
    iso_control = "N/A"
    if is_anomaly and predicted_class_name in GRC_MAPPING:
        nist_control = GRC_MAPPING[predicted_class_name]["nist"]
        iso_control = GRC_MAPPING[predicted_class_name]["iso"]
        
    return {
        "status": "evaluated",
        "zero_day_anomaly_probability": round(anomaly_prob, 4),
        "is_anomaly": is_anomaly,
        "forensic_label": predicted_class_name,
        "remediation_status": remediation_status,
        "remediation_action": action_taken,
        "shap_explainability": [round(val, 4) for val in shap_explainability],
        "compliance_auditing": {
            "nist_control": nist_control,
            "iso_control": iso_control
        }
    }
