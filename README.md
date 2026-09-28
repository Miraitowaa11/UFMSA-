# UFMSA

Official implementation of **UFMSA (Uncertainty Fusion-Based Multimedia
Semantic Alignment)** for multimodal fake news detection.

This repository provides the implementation of the proposed
uncertainty-aware multimodal fusion framework, including:

-   Symmetric Multimodal Co-Attention (SMCA)
-   Sample-adaptive Mixture-of-Experts (MoE)
-   Hierarchical Uncertainty-aware Fusion Network (HUFN)
-   Uncertainty-Adaptive Fusion (UAF)

The complete implementation is provided in `UFMSA.py`.

------------------------------------------------------------------------

## 1. Framework Overview

UFMSA jointly models three modalities:

-   Text modality
-   Visual modality
-   Frequency-domain modality

The framework contains:

1.  Multimodal feature extraction
2.  Symmetric cross-modal semantic interaction with SMCA
3.  Sample-adaptive multimodal association modeling with MoE
4.  Hierarchical uncertainty estimation
5.  Uncertainty-aware adaptive fusion

------------------------------------------------------------------------

## 2. Pretrained Encoders

The framework uses:

-   RoBERTa for textual representation extraction
-   Vision Transformer (ViT) for visual representation extraction
-   S-Transform based encoder for frequency-domain representation
    extraction

All modality representations are projected into a unified
512-dimensional feature space before multimodal interaction.

------------------------------------------------------------------------

## 3. Implementation Details

### 3.1 SMCA

SMCA models symmetric interactions among text, visual, and
frequency-domain modalities.

The implementation contains six directional cross-modal interactions:

-   text-to-frequency
-   text-to-visual
-   frequency-to-text
-   frequency-to-visual
-   visual-to-text
-   visual-to-frequency

Each interaction uses modality-specific query, key, and value
projections followed by scaled dot-product attention.

The implementation follows the computational procedure described in the
manuscript.

------------------------------------------------------------------------

### 3.2 MoE

The MoE module receives SMCA-enhanced modality descriptors and pairwise
cosine similarity cues.

The routing representation is:

\[ x=\[F_t',F_d',F_v',sim(t,d),sim(t,v),sim(d,v)\] \]

Each modality descriptor has dimension 512, and three scalar similarity
values are appended:

\[ d_x=3`\times512`{=tex}+3=1539 \]

The gating network generates sample-specific expert weights for adaptive
multimodal interaction modeling.

The default configuration uses three experts.

------------------------------------------------------------------------

### 3.3 Uncertainty-aware Fusion

UFMSA estimates uncertainty from two complementary perspectives:

**Feature-level uncertainty**

Feature-level uncertainty is estimated through stochastic forward passes
with MC Dropout and measures the variability of learned modality
representations.

**Decision-level uncertainty**

Decision-level uncertainty is estimated from modality-specific
prediction distributions using predictive entropy.

The uncertainty estimates are used to derive confidence-aware modality
weights and adaptively balance discriminative and reliability-oriented
fusion pathways.

------------------------------------------------------------------------

## 4. Environment

The experiments were conducted with:

-   Python 3.x
-   PyTorch 1.11.0 + CUDA 11.3
-   torchvision 0.12.0
-   transformers 4.46.3
-   numpy 1.24.3
-   scikit-learn 1.3.2

Install dependencies:

``` bash
pip install -r requirements_UFMSA.txt
```

------------------------------------------------------------------------

## 5. Dataset Format

The implementation supports JSON, JSONL, and CSV manifest files.

Each sample should contain:

``` json
{
  "text": "sample text",
  "image_path": "path/to/image",
  "label": 0
}
```

Required fields:

-   text
-   image_path
-   label

------------------------------------------------------------------------

## 6. Dataset Split

The experiments use a fixed stratified split:

-   Training: 72%
-   Validation: 8%
-   Testing: 20%

------------------------------------------------------------------------

## 7. Experimental Settings

  Parameter            Value
  -------------------- ------------------------
  Model dimension      512
  Attention heads      8
  Number of experts    3
  MC dropout samples   20
  Batch size           16
  Optimizer            AdaBelief
  Learning rate        1e-4
  Weight decay         0.15
  Dropout              0.5
  Maximum epochs       100
  Early stopping       Patience = 10
  Random seeds         13, 21, 43, 2026, 3407

------------------------------------------------------------------------

## 8. Running

Configure dataset paths and pretrained model paths according to your
local environment, then run:

``` bash
python UFMSA.py
```

------------------------------------------------------------------------


For reproducible experiments, please ensure:

-   Required dependencies are installed.
-   Dataset manifests follow the required format.
-   Pretrained encoder paths are correctly configured.
-   Experimental settings remain consistent with the manuscript.
