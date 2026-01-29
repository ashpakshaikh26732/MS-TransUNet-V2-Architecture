# MS-TransUNet V2: Solving Z-Axis Collapse in 3D Medical Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c)](https://pytorch.org/)
[![Medical AI](https://img.shields.io/badge/Task-3D%20Segmentation-blue)](https://miccai.org/)

**MS-TransUNet V2** is a multi-scale Transformer-UNet architecture specifically designed to recover lost 3D geometry in anisotropic (thick-slice) CT/MRI volumes by introducing a *stride-aware global context bridge* between shallow geometric features and deep semantic representations.

---

## 🧠 The Problem: Z-Axis Collapse in 3D CNNs

Medical volumes are rarely isotropic. Typical CT spacing:  
`0.7mm × 0.7mm × 5.0mm`

Standard 3D U-Nets apply isotropic pooling `(2,2,2)` which causes:

| Stage | Z-Slices |
|-------|----------|
| Input | 32 |
| Down1 | 16 |
| Down2 | 8 |
| Down3 | 4 |
| Bottleneck | 2 ❌ |

By the bottleneck, 3D context collapses into a 2D "pancake".  
Transformers placed here receive almost no geometric information.

Result:  
• Blurry organ shapes  
• Disconnected masks  
• Poor inter-slice coherence  

![Standard TransUNet Failure](Architecture%20Diagrams/old_transunet.png)

---

## 💡 The Solution: MS-TransUNet V2

### 1. Multi-Scale Tokenization
Tokens are extracted from all encoder stages:

| Level | Feature | Role |
|------|---------|------|
| f1,f2 | High-res | Geometry, boundaries |
| f3 | Mid-level | Structural continuity |
| f4 | Low-res | Global semantics |

All are projected into a shared latent space and concatenated into a single token sequence.

---

### 2. Stride-Aware 3D Positional Encoding (Core Innovation)

Each scale uses continuous 3D sinusoidal embeddings scaled by **network stride**:

\[
pos_scaled = index * 2^{level}
\]

This ensures physical alignment:

- Token (1,1,1) at Level-3 (stride 8)  
- Physically corresponds to region (8,8,8) at Level-0  

Thus the Transformer attends in **millimeter space**, not index space.

---

### 3. Global Semantic Bridge

The joint token sequence enables:

- Deep semantic tokens (f4) to attend to
- Shallow geometric tokens (f1,f2)
- With correct physical alignment

This restores 3D topology and long-range organ continuity.

![MS-TransUNet V2 Architecture](Architecture%20Diagrams/MS-TransUNet%20V2%20-%20Multi%20Scale%20Transformer%20UNet%203D.png)





## 🚀 Usage

```python
import torch
from MS_TransUNet_V2_code import Transunet_V2

model = Transunet_V2(in_channels=1, num_classes=2).cuda()
x = torch.randn(1, 1, 96, 96, 96).cuda()
y = model(x)

print(y.shape)  # (1, 2, 96, 96, 96)
