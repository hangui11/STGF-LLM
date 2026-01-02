# Synthetic Time-Series Generation and Forecasting Using Large Language Models (STGF-LLM)

This repository contains the official implementation, datasets, and code for **STGF-LLM**, a unified framework utilizing Large Language Models (LLMs) for both time-series forecasting and synthetic data generation.

## 📌 Project Overview

This project benchmarks a novel LLM-based approach against traditional statistical methods (ARIMA) and state-of-the-art Deep Generative Models (TimeGAN, DoppelGANger, Chronos-T5).

**Key Features:**

* **Unified Architecture:** Handles both forecasting and synthetic generation.
* **History-Aware Prompting:** Novel prompt design for time-series serialization.
* **Comprehensive Benchmarking:** Includes code for ARIMA, TimeGAN, DoppelGANger, and Chronos-T5.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

This project utilizes **Git LFS** (Large File Storage) for model weights and datasets. Ensure you have Git LFS installed before cloning.

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone the repository
git clone https://github.com/hangui11/STGF-LLM.git
cd STGF-LLM

# Pull large files
git lfs pull

```

### 2. Environment Setup (Critical)

Due to conflicting dependencies between legacy GAN libraries (TensorFlow 1.x/Compat) and modern LLM libraries (PyTorch/Transformers), **two separate Conda environments are required.**

#### 🟢 Environment A: For GAN Models (TimeGAN, DoppelGANger)

Use this environment for `2. timeGAN.ipynb` and `2. doppelGANger.ipynb`.

```bash
conda create -n ts-gan python=3.10.19
conda activate ts-gan

# Install dependencies
pip install y-data-synthetic tensorflow numpy<2.0.0 pandas

```

#### 🔵 Environment B: For LLMs, ARIMA & Evaluation

Use this environment for `1. preprocess.ipynb`, `2. stgf_llm.ipynb`, `2. chronos_t5.ipynb`, `2. arima.ipynb`, and `3. evaluation.ipynb`.

```bash
conda create -n ts-llm python=3.11.0
conda activate ts-llm

# Install dependencies
pip install scikit-learn pandas numpy pmdarima torch transformers peft tqdm datasets accelerate chronos-forecasting huggingface
```

---

## 📂 Repository Structure

```text
├── 📁 data/                        # Contains all datasets
│   ├── {Dataset_Name}/
│   │   ├── original.csv           # Raw data
│   │   ├── *_llm_text.csv         # Serialized text data for STGF-LLM
│   │   ├── *_numerical.csv        # Numerical data for baselines
│   │   ├── arima_forecasting.csv  # Generated output (Example)
│   │   ├── timegan_synthetic.csv  # Generated output (Example)
│   │   └── ...                    # All other generated files are here
│
├── 📁 doppelganger_models/         # Pre-trained DoppelGANger checkpoints
├── 📁 timegan_models/              # Pre-trained TimeGAN checkpoints
├── 📁 stgf_llm_adapters/           # LoRA adapters for STGF-LLM
├── 📁 scaler/                      # Saved Scaler objects for numerical inversion
├── 📁 images/                      # t-SNE visualization plots
│
├── 📓 1. preprocess.ipynb          # Data cleaning & serialization
├── 📓 2. arima.ipynb               # Baseline: ARIMA Forecasting
├── 📓 2. chronos_t5.ipynb          # Baseline: Chronos-T5 Foundation Model
├── 📓 2. doppelGANger.ipynb        # Baseline: DoppelGANger Generation (Use Env A)
├── 📓 2. timeGAN.ipynb             # Baseline: TimeGAN Generation (Use Env A)
├── 📓 2. stgf_llm.ipynb            # Proposed Method: STGF-LLM (Training & Inference)
└── 📓 3. evaluation.ipynb          # Metrics, TSTR Tables & Visualization Generation

```

---

## 🚀 Usage Workflow

Please follow the notebooks in the numbered order:

1. **Data Preparation:**
    * Run `1. preprocess.ipynb` (Env B) to clean data, normalize numerical values, and generate text-serialized versions for the LLM.


2. **Model Training & Inference:**
    * **Baselines:** Run the respective notebooks for ARIMA, TimeGAN, or DoppelGANger.
    * **Proposed Method:** Run `2. stgf_llm.ipynb` (Env B). This notebook handles:
        * Loading the Qwen backbone.
        * Fine-tuning with LoRA.
        * Autoregressive generation for forecasting and synthesis.




3. **Evaluation:**
    * Run `3. evaluation.ipynb` (Env B) to generate comparison tables and visualizations.
* **Note:** Ensure `doppelganger_models` or `timegan_models` are available if you wish to evaluate those baselines without retraining.



---

## 📊 Results & Visualization

* **t-SNE Plots:** The `images/` folder contains the visual projections comparing the distribution fidelity of real vs. synthetic data.
* **TSTR & Metrics:** All quantitative results, including **RMSE**, **MAPE**, and **TSTR (Training on Synthetic, Testing on Real)** tables, are generated and displayed directly within `3. evaluation.ipynb`.

---

## ⚠️ Troubleshooting

* **DoppelGANger Checkpoint Error:** If you encounter `Couldn't match files for checkpoint`, open the `checkpoint` file inside the model folder and ensure the path is set to relative: `model_checkpoint_path: "doppelganger"`.
* **PEFT Adapter Warning:** If you see `UserWarning: You are trying to modify a model with PEFT for a second time`, restart the kernel to clear the model from memory before re-running.