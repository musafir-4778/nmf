# Deep Matrix Factorization for Attribute Representation

## 👥 Group Members

| Name             | Roll Number |
|------------------|-------------|
| Anubhav Singh    | 202251018   |
| Ashish Tetarwal  | 202251024   |
| Lakshya Yadav    | 202251067   |
| Riyank Singh     | 202251127   |

---

## 🎯 Goal
This project is a reimplementation of the paper:  
**“A Deep Matrix Factorization Method for Learning Attribute Representations”**.
Paper Link: https://arxiv.org/abs/1509.03248

We applied the proposed **deep semi-NMF + graph regularization pipeline** to the **CelebA dataset**.

---

## 🛠 What We Did

### 1. Dataset Preprocessing
- **Dataset**: Used the CelebA dataset (~200,000 celebrity face images).
- **Image Handling**:
    - Converted images to grayscale
    - Resized to `64×64`
    - Flattened into vectors for processing
- **Binary Attributes Extracted**:
    - **Layer 1 (H1):** Young
    - **Layer 2 (H2):** Smiling

### 2. Graph Construction
- **Method**: Built **k-Nearest Neighbors (kNN) Laplacians (W, D)** for both the *Young* and *Smiling* attributes.
- **Similarity Metric**: Cosine similarity
- **Hardware**: NVIDIA RTX 4060 GPU
- **Scale**: Each graph ≈ **3 million edges**

### 3. Model Training
- **Pre-training**: Layer-wise semi-NMF for initialization
- **Fine-tuning**: Joint deep semi-NMF with **graph regularization**
- **Implementation**: PyTorch with CUDA acceleration

---

## 📊 Evaluation & Results

### Validation Accuracy
- **Young (H1):** `0.747` (~75%)
- **Smiling (H2):** `0.517` (~52%)

Evaluation was performed by aligning the learned H1/H2 factors with the ground-truth attributes using **Logistic Regression**.

### Runtime Performance (on RTX 4060 GPU)
- **Graph Construction:** ~17.5 minutes
- **Pre-training:** ~7.5 minutes
- **Fine-tuning:** ~24 minutes
- **Total Pipeline:** ~49 minutes

---

## 🔑 Key Observations
- **Attribute Disentanglement**
    - H1 captured **age-related variations**
    - H2 captured **expression variations (smiling)**

- **Bottlenecks**
    - Graph construction = primary performance bottleneck
    - Using **approximate kNN (e.g., FAISS)** could significantly speed this up

- **Numerical Stability**
    - Ridge regularization and NaN guards were added in factor update rules

---

## ✅ Summary
We successfully implemented the **deep semi-NMF pipeline** from the paper and validated its **attribute alignment capabilities** on the CelebA dataset.

The end-to-end system functions as expected and is **extensible for future work**. 🚀

---
