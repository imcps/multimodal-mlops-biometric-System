# Multimodal MLOps Biometric System

This repository presents a **production‑oriented MLOps implementation** of a **multimodal biometric recognition system** using **Iris and Fingerprint images**.

The primary focus of this project is **system design, scalability, and MLOps fundamentals**, rather than model performance or algorithmic novelty.

***

## 🎯 Project Objective

The goal of this project is to demonstrate:

*   Clean, modular, **production‑quality Python code**
*   A full **multimodal machine learning workflow** in PyTorch
*   **Scalable and efficient data preprocessing pipelines**
*   Use of **parallel data processing** for performance
*   **Reproducible training and inference**
*   **Checkpointing** and **experiment tracking**
*   Clear **architectural reasoning and trade‑offs**

This is **not an accuracy‑driven ML project**.

***

## 🚫 Non‑Goals (Intentional)

The following are out of scope by design:

*   Model accuracy optimization
*   Hyperparameter tuning
*   Complex model architectures
*   Statistical evaluation and benchmarking
*   Security‑grade biometric deployment

This aligns with the evaluation criteria, where **architecture and engineering quality** are weighted more than metrics.

***

## 📂 Dataset

**Source:**
Kaggle – *Multimodal Iris & Fingerprint Biometric Dataset*

### Raw Dataset Structure

The dataset is organized per person as follows:

    data/raw/<person_id>/
    ├── Fingerprint/        # Fingerprint images
    │   └── *.bmp
    ├── left/              # Iris (left eye)
    │   └── *.bmp
    └── right/             # Iris (right eye)
        └── *.bmp

Operating‑system artifacts (e.g., `desktop.ini`) are present and handled during preprocessing.

***

## 🧠 System Architecture

    Raw Images (Filesystem)
            ↓
    Parallel Preprocessing (Ray)
            ↓
    Normalized Metadata (Parquet)
            ↓
    PyTorch Datasets (Lazy Image Loading)
            ↓
    Per‑Modality Encoders
            ↓
    Feature‑Level Fusion
            ↓
    Training / Inference Pipelines

### Design Principles

*   **Separation of concerns**
*   **Reproducibility**
*   **Scalability**
*   **Fault‑tolerant data handling**
*   **Infrastructure‑aware design**

***

## 🧩 Repository Structure

    .
    ├── configs/                 # Configuration files
    ├── data/
    │   ├── raw/                 # Raw Kaggle data (git‑ignored)
    │   └── processed/           # Generated Parquet metadata
    ├── scripts/
    │   ├── preprocess.py        # Ray‑based preprocessing
    │   └── benchmark_dataloader.py
    ├── src/
    │   ├── datasets/            # Iris, Fingerprint, Multimodal datasets
    │   ├── models/              # Encoders, fusion, multimodal model
    │   ├── pipelines/           # Training & inference orchestration
    │   └── utils/               # Reproducibility, profiling, checkpoints
    ├── checkpoints/             # Model checkpoints (git‑ignored)
    ├── runs/                    # TensorBoard logs (git‑ignored)
    ├── tests/                   # Lightweight tests
    └── README.md

Each layer is **modular** and **independently extensible**, mirroring real production ML systems.

***

## ⚙️ Data Preprocessing

### Why Preprocessing Is Required

The raw dataset:

*   Uses a non‑canonical folder layout
*   Contains OS‑specific system files
*   Is not directly suitable for ML training

### What Preprocessing Does

*   Traverses the raw filesystem
*   Filters non‑image files
*   Normalizes modality semantics:
    *   `Fingerprint` → `fingerprint`
    *   `left`, `right` → `iris`
*   Validates images safely and in parallel using **Ray**
*   Writes normalized metadata to **Parquet**

### Why Parquet?

*   Columnar and compressed format
*   Schema enforcement
*   Fast filtering and scalable ingestion
*   Cloud and distributed‑system friendly

***

## 🧠 Model Design

*   **Encoders:** One encoder per modality (iris, fingerprint)
*   **Fusion:** Feature‑level concatenation
*   **Prediction Target:** Person ID (classification)

The model architecture is intentionally simple to keep the focus on **system correctness and explainability**, not raw accuracy.

***

## 🔄 Training Pipeline

The training pipeline demonstrates:

*   Deterministic execution (seed control)
*   Lazy data loading via PyTorch `Dataset`
*   Explicit training loop (no hidden abstractions)
*   TensorBoard‑based experiment logging
*   Checkpoint save and resume support

Training runs for a small number of batches per epoch by design.

***

## 🔎 Inference Pipeline

The inference pipeline demonstrates:

*   Loading trained checkpoints
*   Model‑only restoration (optimizer optional)
*   Forward‑pass prediction
*   Deployment‑style execution logic

Inference is used to validate **end‑to‑end pipeline integrity**, not to compute evaluation metrics.

***

## 📊 Experiment Tracking

TensorBoard is used for lightweight experiment tracking:

Tracked signals:

*   Training loss
*   Learning rate
*   Run metadata

Launch TensorBoard:

```bash
tensorboard --logdir=runs
```

***

## 💾 Checkpointing

Each checkpoint includes:

*   Model parameters
*   Optimizer state
*   Epoch index

Enables:

*   Training resumption
*   Model loading for inference
*   Cloud‑backed storage (future extension)

Checkpoint logic is deliberately kept **outside the model code**.

***

## 🔁 Reproducibility

Reproducibility is enforced via:

*   Python random seed
*   NumPy seed
*   PyTorch CPU & CUDA seeds
*   Deterministic CuDNN settings

***

## ▶️ How to Run

### 1️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Place Dataset

Extract Kaggle dataset into:

    data/raw/

### 4️⃣ Preprocess Data

```bash
python scripts/preprocess.py
```

### 5️⃣ Train Model

```bash
python -m src.pipelines.train
```

### 6️⃣ Run Inference

```bash
python -m src.pipelines.inference
```

### 7️⃣ Run Tests

```bash
pytest
```

***

## ☁️ Cloud Readiness

The architecture maps directly to:

*   Azure Blob / S3 for data and checkpoints
*   Kubernetes (AKS/EKS) for training jobs
*   MLflow / AzureML for advanced tracking

No refactor is required for cloud execution.

***

## 🔍 Trade‑offs and Design Decisions

| Decision            | Rationale                             |
| ------------------- | ------------------------------------- |
| Simple model        | Focus on MLOps, not accuracy          |
| No train/test split | Performance evaluation not required   |
| TensorBoard         | Lightweight, low overhead             |
| Ray preprocessing   | Efficient CPU‑bound parallelism       |
| Parquet metadata    | Scalable, reproducible data interface |

***

