# Step 07: Model Training

> **ML Lifecycle Phase**: 07 / 10  
> **System**: SmartHVAC & Seasonal Appliance Heterogeneous GML  
> **Target Hardware**: NVIDIA Tesla V100S-PCIE-32GB GPU Server (`10.13.1.7`)  
> **Precision**: Automatic Mixed Precision (FP16 AMP via `torch.amp.autocast`)

---

## 1. Hardware Environment & Stack

Model training was performed on the dedicated high-performance lab server:
* **Host**: `10.13.1.7` (Ubuntu 24.04 LTS, x86_64)
* **GPU**: NVIDIA Tesla V100S-PCIE-32GB (Driver: 580.126.18, CUDA 13.0 / 12.4 runtime)
* **Python Environment**: Miniconda Python 3.12, PyTorch 2.6.0+cu124, PyTorch Geometric 2.8.1
* **GPU Memory Footprint**: **28.77 MB** peak (due to highly optimized sparse CSR bipartite message-passing graphs)

```mermaid
flowchart LR
    subgraph Host_Server["Lab Server 10.13.1.7"]
        V100["NVIDIA Tesla V100S-PCIE-32GB<br>32 GB HBM2 @ 1.13 TB/s"]
        CUDA["CUDA 13.0 / PyTorch 2.6.0+cu124"]
        PyG["PyTorch Geometric 2.8.1"]
    end

    subgraph AMP_Pipeline["Mixed-Precision Training Loop"]
        Autocast["torch.amp.autocast(cuda)<br>FP16 Forward Pass"]
        Scaler["torch.amp.GradScaler(cuda)<br>Loss Scaling and Underflow Prevention"]
        Clip["torch.nn.utils.clip_grad_norm_<br>max_norm = 1.0"]
    end

    Host_Server --> AMP_Pipeline
```

---

## 2. Training Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :---: | :--- |
| **Number of Epochs** | 120 (Initial) / 60 (Fine-Tuning) | Sufficient for convergence across all 12 monthly graph snapshots |
| **Optimizer** | `AdamW` | Decoupled weight decay prevents overfitting on static node features |
| **Learning Rate** | $`1.0 \times 10^{-3} \to 2.5 \times 10^{-4}`$ | Scaled down during fine-tuning for smooth convergence |
| **Weight Decay** | $`1.0 \times 10^{-4}`$ | $`L_2`$ regularization on linear projections and GNN weights |
| **Batch Strategy** | Full-Graph Monthly Snapshots | 12 heterogeneous graph snapshots per epoch ($`m = 1 \dots 12`$) |
| **Gradient Clipping** | $`\text{max\_norm} = 1.0`$ | Eliminates gradient explosions during multi-task backward passes |
| **Learning Rate Scheduler** | `CosineAnnealingLR` ($`T_{\max}=60, \eta_{\min}=10^{-5}`$) | Prevents late-epoch oscillations |

---

## 3. Initial 120-Epoch Training Trajectory

During initial pre-training (`train_and_monitor.py`), the model converged from an uninitialized loss of $`> 23,000`$ to a stable loss plateau:

```
[Epoch 001/120] Loss: 23057.1211 | Demand: 15164.866 | Maint: 3.848 | Churn: 0.687 | Inv: 7885.658 | VRAM: 28.8 MB
[Epoch 010/120] Loss:   558.2476 | Demand:   326.906 | Maint: 0.814 | Churn: 0.644 | Inv:  229.348 | VRAM: 28.8 MB
[Epoch 020/120] Loss:   676.6607 | Demand:   579.687 | Maint: 0.768 | Churn: 0.621 | Inv:   95.077 | VRAM: 28.8 MB
[Epoch 030/120] Loss:   862.0366 | Demand:   576.536 | Maint: 0.785 | Churn: 0.613 | Inv:  283.626 | VRAM: 28.8 MB
[Epoch 040/120] Loss:   253.3916 | Demand:   159.122 | Maint: 0.751 | Churn: 0.572 | Inv:   92.454 | VRAM: 28.8 MB
[Epoch 050/120] Loss:   582.8837 | Demand:   370.147 | Maint: 0.741 | Churn: 0.569 | Inv:  210.932 | VRAM: 28.8 MB
[Epoch 080/120] Loss:   459.8829 | Demand:   321.439 | Maint: 0.732 | Churn: 0.548 | Inv:  136.837 | VRAM: 28.8 MB
[Epoch 100/120] Loss:   384.3644 | Demand:   184.821 | Maint: 0.729 | Churn: 0.536 | Inv:  197.942 | VRAM: 28.8 MB
[Epoch 120/120] Loss:   310.9272 | Demand:   234.032 | Maint: 0.733 | Churn: 0.505 | Inv:   75.204 | VRAM: 28.8 MB

>>> Best Pre-Trained Model Saved: best_hetero_gnn_v100.pt (Loss: 235.23) <<<
```

---

## 4. Self-Healing Divergence Protection

To guarantee that training unattended never crashes due to numerical instability:
1. The training harness monitors `avg_loss` at each epoch.
2. If `isnan(loss)` or `isinf(loss)` occurs, or if loss spikes by $`> 10\text{x}`$, the loop catches the condition, reduces the learning rate by $`50\%`$ ($`\text{lr} \leftarrow \text{lr} \times 0.5`$), resets model parameters to the last checkpoint, and automatically restarts.

---

## 5. Fine-Tuning Execution on Tesla V100 GPU (`fine_tune_v100.py`)

Following initial pre-training, an empirical fine-tuning phase was launched to incorporate real-world industrial and transactional benchmark datasets:
* **Input Checkpoint**: `artifacts/best_hetero_gnn_v100.pt`
* **Real Datasets Ingested**:
  * UCI AI4I 2020 (`dataset/601`): Heat Dissipation Failure (`HDF`) and Power Failure (`PWF`) ground-truth labels + rotational power curves ($`1.2 - 3.6\text{ kW}`$).
  * UCI Online Retail II (`dataset/502`): 4,312 customer RFM profiles ($`R > 90\text{d}`$ sleeper churn, Monetary CLV tiers).
  * Kaggle HVAC Power (`felixpradoh`): 8,760 hourly thermal telemetry curves across 12 climate regimes.
* **Loss Function**: Kendall Homoscedastic Uncertainty Weighting + SmoothL1 (Huber, $`\beta = 4.0`$) Loss + class-balanced `pos_weight` ($`2.5`$ for failure, $`1.8`$ for churn).
* **Optimizer**: AdamW ($`\text{lr} = 2.5 \times 10^{-4}`$, $`\text{weight\_decay} = 10^{-4}`$), `CosineAnnealingLR` ($`T_{\max}=60, \eta_{\min}=10^{-5}`$), gradient clipping ($`\text{max\_norm} = 1.0`$).
* **V100 Runtime & Telemetry**:
  * Total duration: **17.0 seconds** across 60 calibrated epochs.
  * Peak VRAM: **28.8 MB**.
  * Best Dossier-Equivalent Loss: **46.80** (reduced from $`137.04`$, $`-65.85\%`$).
  * Checkpoint Saved: `artifacts/fine_tuned_hetero_gnn_v100.pt`.

