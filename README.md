# Modeling Energy Consumption in Deep Learning Architectures Using Power Laws

This repository contains the code, data, and analysis accompanying the research paper:

> **Modeling Energy Consumption in Deep Learning Architectures Using Power Laws**  
> _Mansour MAYAKI_  
> _ECAI 2025_  
> [DOI or arXiv link]


---

## 📚 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Practical Use Case: Energy-Aware Model Selection](#-practical-use-case-energy-aware-model-selection)
- [Datasets](#-datasets)
- [Results](#-results)
- [License](#-license)
- [Citation](#-citation)
- [Contributions](#-contributions)

---
---

## 📄 Overview

This paper bridges this gap by conducting a detailed empirical investigation into the energy efficiency of three widely used model architectures: LSTM, GRU, and Transformers. These architectures are fundamentally different in their computational design. LSTMs and GRUs are recurrent neural networks (RNNs) optimized for processing sequential data through memory mechanisms, while Transformers employ self-attention mechanisms, enabling superior performance on tasks involving long-range dependencies. Despite these differences, all three architectures are widely used in applications ranging from NLP to time series forecasting, making them ideal candidates for a comparative energy consumption study.
Our study focuses on measuring the energy consumption of these architectures during training under varying model configurations, such as the number of layers, hidden dimensions, and attention heads (for Transformers). Beyond raw energy measurements, we also explore the relationship between energy usage and model performance metrics, such as hardware efficiency and the number of floating point operations (FLOPs). Our goal is to show that it is possible to derive a general law that makes energy consumption predictable, given a model architecture.
It includes:
- Preprocessing scripts for data
- Training and evaluation of [your models]
- Code to reproduce figures and tables from the paper
- A small, publicly available dataset on [topic]

---

## 🗂 Repository Structure

```bash
.
├── data/               # Raw and processed data
├── notebooks/          # Jupyter notebooks with analysis and experiments
├── src/                # Python scripts for preprocessing, training, etc.
├── paper/              # LaTeX source files of the paper
├── results/            # Output: figures, tables, metrics
├── requirements.txt    # Python dependencies
├── environment.yml     # (optional) Conda environment
├── LICENSE             # License for code and/or data
└── README.md           # This file


---

## 📦 Installation

### Option 1: Using `pip`

```bash
git clone https://gitlab.com/yourusername/yourproject.git
cd yourproject
pip install -r requirements.txt
```

## 🔋 Practical Use Case: Energy-Aware Model Selection

This section presents a practical use of our energy estimation method to support model selection based on energy efficiency.  
We compare two Transformer configurations on a GPU (**NVIDIA A100**, max throughput **156 TFLOPs/s**).  

Rather than benchmarking, our method uses architectural parameters to estimate energy consumption.

### Hardware and Input Settings
All steps are encapsulated in modular functions—define your model configuration and workload parameters (batch size, sequence length, layers, heads, embedding dimensionality $d_{\text{model}}$), and the system automatically computes the estimated energy consumption for one training epoch.

- **Max Throughput:** $v_{\text{max}} = 156 \times 10^{12}$ FLOPs/s  
- **Batch Size:** 64  
- **Sequence Length:** 320  

### Model Configurations
- **Model A:** 6 layers, $d_{\text{model}} = 512$, 8 attention heads  
- **Model B:** 12 layers, $d_{\text{model}} = 768$, 12 attention heads  

---

### **Step 1 — FLOPs Estimation**

| Operation         | Model A FLOPs          | Model B FLOPs          |
|-------------------|------------------------|------------------------|
| QKV Projections   | $6.44 \times 10^{10}$  | $3.43 \times 10^{11}$  |
| Final Projection  | $4.30 \times 10^{10}$  | $2.29 \times 10^{11}$  |
| Attention Scores  | $6.74 \times 10^{9}$   | $2.69 \times 10^{10}$  |
| Attention Output  | $6.74 \times 10^{9}$   | $2.69 \times 10^{10}$  |

---

### **Step 2 — Hardware Efficiency & Duration Estimation**

| Operation         | $\eta_{\text{max}}$ | $k$   | $\alpha$ |
|-------------------|--------------------|-------|----------|
| Projections       | 65.79              | 9.25  | 0.81     |
| Attention Scores  | 55.6               | 8.24  | 0.80     |
| Attention Output  | 68.22              | 8.49  | 0.794    |

| Operation         | Model | FLOPs (TF) | $\eta_\Theta(c)$ | $t_\Theta(c) (\mu s) $|
|-------------------|-------|------------|------------------|------------------------|
| Projections       | A     | 0.1074     | 35.45            | 19.4                   |
| Attention Scores  | A     | 0.00674    | 5.15             | 0.76                   |
| Attention Output  | A     | 0.00674    | 6.68             | 0.64                   |
| Projections       | B     | 0.572      | 57.65            | 63.6                   |
| Attention Scores  | B     | 0.0269     | 11.98            | 1.23                   |
| Attention Output  | B     | 0.0269     | 15.37            | 1.12                   |

---

### **Step 3 — Energy Estimation**

Regression model:

$$
E = 23.8962 + 0.1089 \cdot t_{\text{Proj}} + 0.4743 \cdot t_{\text{Score}} - 0.1029 \cdot t_{\text{Mul}}
$$

For Model A:

$$
E_A \approx 23.8962 + 0.1089 \cdot 19.4 + 0.4743 \cdot 0.76 - 0.1029 \cdot 0.64 = \mathbf{26.30\ J}
$$

For Model B:

$$
E_B \approx 23.8962 + 0.1089 \cdot 63.6 + 0.4743 \cdot 1.23 - 0.1029 \cdot 1.12 = \mathbf{31.28\ J}
$$

---

### **Conclusion**
Under identical conditions, Model B consumes **19% more energy per epoch** than Model A. Over one million epochs, this difference accumulates to **1.3 kWh**, which is significant in battery-powered or high-throughput systems.


---

## ▶️ How to Run

### 1. Preprocess Data

```bash
python src/preprocess.py
```

### 2. Train Model

```bash
python src/train_model.py
```

### 3. Reproduce Results

All figures and tables can be regenerated using the notebooks in `notebooks/`.

---

## 📊 Datasets

* 📂 `data/raw/`: Original measurements
* 📂 `data/processed/`: Cleaned datasets for modeling

> ⚠️ Due to size or license restrictions, some datasets may not be included directly. Please follow the instructions in `data/README.md` to download them.

---

## 📈 Results

Key results can be found in the `results/` folder, including:

* Plots and metrics
* LaTeX-formatted tables for publication

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
Please cite our work if you use any part of this project.

---

## 📚 Citation

```bibtex
@article{your2025paper,
  title={Your Paper Title},
  author={Author, A. and Researcher, B.},
  journal={Journal of Awesome ML Research},
  year={2025}
}
```

---

## 🤝 Contributions

Pull requests and issues are welcome. Please feel free to fork and adapt this work for your own research!

```



